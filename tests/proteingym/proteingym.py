# list all files in /data2/malbrank/protein_gym/DMS_msa_files by order of size
import os
import random

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mamba_ssm.utils.generation import *
from tqdm import tqdm

from scripts import Uniclust30_Dataset, MambaLMHeadModelwithPosids, load_model, load_from_file, tokenizer, \
    prepare_target, prepare_tokens, generate_sequence, AA_TO_ID
from poet_sampling import MSASampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_USE_CUDA_DSA'] = "0"

_checkpoints = {"mamba": "/nvme1/common/mamba_100M_FIM_checkpoint_32k-100000",
                "mamba_ft": "/nvme1/common/mamba_100M_FIM_finetuned_32k_checkpoint-17000"}


def mutational_landscape(model, msa, num_sequences=50, mutated_pos=None, batch_size=16):
    """Generate the mutational landscape of a single protein sequence.
    Args:
        model (MambaLMHeadModelwithPosids): the model to use for generation.
        msa (list): a list of sequences in the MSA.
        num_sequences (int): the number of sequences to sample from the MSA.
        mutated_pos (list): a list of positions that have been mutated.
        batch_size (int): the batch size to use for generation.
        """
    target_sequence = msa[:1]
    context_msa = msa[1:][::-1]
    all_mut_effects = []
    all_logits = []
    sequence_length = len(target_sequence[0])

    for i in tqdm(range(0, sequence_length, batch_size), desc="Generating mutational landscape"):
        torch.cuda.empty_cache()
        # Tokenize the sequences and concatenate them into a single array
        target = tokenizer(target_sequence, concatenate=True)
        tokens = tokenizer(context_msa, concatenate=True)
        batch_mut_effects = []
        batch_logits = []
        for j in range(i, min(i + batch_size, sequence_length)):
            if j not in mutated_pos:
                batch_mut_effects.append(torch.zeros(20).cpu())
                continue

            input_seq, targ_pos, is_fim_dict = prepare_target(target, use_fim={"<mask-1>": ((j + 1, j + 2), 1)})
            context_tokens, context_pos_ids = prepare_tokens(tokens,
                                                             target_tokens=input_seq,
                                                             target_pos_ids=targ_pos,
                                                             DatasetClass=Uniclust30_Dataset,
                                                             num_sequences=num_sequences,
                                                             fim_strategy="multiple_span",
                                                             mask_fraction=0,
                                                             max_patches=0,
                                                             add_position_ids="1d")

            output = generate_sequence(model,
                                       context_tokens,
                                       position_ids=context_pos_ids,
                                       is_fim=is_fim_dict,
                                       max_length=1,
                                       temperature=1.,
                                       top_k=3,
                                       top_p=0.0,
                                       return_dict_in_generate=True,
                                       output_scores=True,
                                       eos_token_id=AA_TO_ID["<cls>"],
                                       device="cuda")
            logits = output["scores"]
            gt = target[0, j + 1].int().item()
            logits_mut = logits[0, 0]
            mut_effect = logits_mut - logits_mut[gt]
            mut_effect = mut_effect[4:24]
            batch_mut_effects.append(torch.tensor(mut_effect).cpu())
            batch_logits.append(torch.tensor(logits_mut).cpu())
        all_mut_effects.extend(batch_mut_effects)
        all_logits.extend(batch_logits)
    all_mut_effects = torch.stack(all_mut_effects, 0)
    all_logits = torch.stack(all_logits, 0)
    return all_mut_effects, all_logits


def evaluate_all_landscapes(checkpoint, prefix, n_tokens_, csv_folder, msa_folder, out_folder, database_df):
    r"""
    Evaluate all spearmanr correlation for every landscape

    Args:
        checkpoint (str): path to the checkpoint
        prefix (str): prefix for the output folder
        n_tokens_ (int): number of tokens
        csv_folder (str): path to the csv folder
        msa_folder (str): path to the msa folder
        out_folder (str): path to the output folder
        database_df (pd.DataFrame): dataframe containing the database
    """
    if not os.path.exists(f"{out_folder}/{prefix}_{n_tokens_}"):
        os.mkdir(f"{out_folder}/{prefix}_{n_tokens_}")

    csv_files = os.listdir(csv_folder)
    csv_files = [f for f in csv_files if f.endswith(".csv")]
    csv_files_to_prot = {}
    for filename in csv_files:
        prot = "_".join(filename.split("_")[:2])
        csv_files_to_prot[filename] = prot

    model = load_model(checkpoint,
                       # /nvme1/common/mamba_100M_FIM_finetuned_32k_checkpoint-16500 "/nvme1/common/mamba_100M_FIM_checkpoint_32k-54500"
                       model_class=MambaLMHeadModelwithPosids,
                       device="cuda",
                       dtype=torch.bfloat16).eval()

    res = {}
    for i, row in database_df.iterrows():
        name = row["DMS_id"]
        csv_filename = row["DMS_filename"]
        msa_start = row["MSA_start"] - 1
        msa_len = row["MSA_len"]
        print("Processing:", name)
        torch.cuda.empty_cache()
        if "indels" in csv_filename:
            continue
        if os.path.exists(msa_folder + csv_files_to_prot[csv_filename] + ".a3m", ):
            msa = load_from_file(msa_folder + csv_files_to_prot[csv_filename] + ".a3m", )
        else:
            msa = load_from_file(msa_folder + csv_files_to_prot[csv_filename].split("_")[0] + ".a3m", )
        msa = [seq[msa_start:msa_start + msa_len] for seq in msa]
        first_seq = msa[0]
        msa_length = n_tokens_ // len(first_seq)
        print("Processing:", name, " with length:", msa_length)
        """max_similarity = .95
        max_dissimilarity = 0.
        sampler = MSASampler(
            max_similarity=max_similarity,
            max_dissimilarity=max_dissimilarity,
        )
        sample_idxs = sampler.get_sample_idxs(msa=msa,)
        sample_idxs = np.random.permutation(sample_idxs)
        sample_idxs = sample_idxs[:msa_length]
        sample_idxs = np.sort(sample_idxs)
        msa = [first_seq] + [msa[i] for i in sample_idxs]"""
        msa = [first_seq] + msa[:msa_length]

        df = pd.read_csv(csv_folder + csv_filename)
        gt_mut_landscape = torch.ones((len(msa[0]), 20)) * np.inf
        single_mutation_only = True
        for i, row in df.iterrows():
            if ":" in row["mutant"]:
                single_mutation_only = False
                continue
            mut_pos = int(row["mutant"][1:-1]) - (msa_start + 1)
            mut_aa = row["mutant"][-1]
            eff = float(row["DMS_score"])
            mut_aa_id = AA_TO_ID[mut_aa] - 4
            gt_mut_landscape[mut_pos, mut_aa_id] = eff
        keep_idx = torch.where(gt_mut_landscape != np.inf)
        mutated_pos = list(torch.where(gt_mut_landscape.min(1).values != np.inf)[0].numpy())
        if f"{name}_landscape.pt" in os.listdir(f"{out_folder}/{prefix}_{n_tokens_}"):
            landscape = torch.load(f"{out_folder}/{prefix}_{n_tokens_}/{name}_landscape.pt")
        else:
            landscape, _ = mutational_landscape(model, msa, len(msa) - 1, mutated_pos)
            torch.save(landscape, f"{out_folder}/{prefix}_{n_tokens_}/{name}_landscape.pt")
        gt_mut_landscape_keep = gt_mut_landscape[keep_idx]
        landscape_keep = landscape[keep_idx]

        sp = spearmanr(gt_mut_landscape_keep, landscape_keep)[0]
        print("Dataset:", name, "Spearman:", sp, "Single mutation only:", single_mutation_only)
        torch.save(landscape, f"{out_folder}/{prefix}_{n_tokens_}/{name}_landscape.pt")
        res[name] = sp

    torch.save(res, f"{out_folder}/{prefix}_{n_tokens_}/spearman.pt")


if __name__ == "__main__":
    n_tokens = [4000, 8000, 16000, 32000, 64000]
    DATA_DIR = "/data2/malbrank"
    csv_folder = f"{DATA_DIR}/protein_gym/substitutions/DMS_ProteinGym_substitutions/"
    msa_folder = f"{DATA_DIR}/protein_gym/mmseqs_colabfold_protocol/"
    out_folder = f"{DATA_DIR}/protein_gym/mut_effects/"
    database_df = pd.read_csv("/data2/malbrank/protein_gym/substitutions/DMS_substitutions.csv")
    prefixes = ["mamba_ft_colabfold_msa", "mamba_colabfold_msa"]
    checkpoints = ["/nvme1/common/mamba_100M_FIM_finetuned_32k_checkpoint-17000",
                   "/nvme1/common/mamba_100M_FIM_checkpoint_32k_checkpoint-100000"]
    for checkpoint, prefix in zip(checkpoints, prefixes):
        for n_tokens_ in n_tokens:
            evaluate_all_landscapes(checkpoint, prefix, n_tokens_, csv_folder, msa_folder, out_folder, database_df)

import os
import sys
from sklearn.metrics import roc_auc_score, matthews_corrcoef

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mamba_ssm.utils.generation import *
from tqdm import tqdm

from ProtMamba_ssm import MambaLMHeadModelwithPosids, load_model, load_from_file, tokenizer, \
    prepare_target, prepare_tokens, generate_sequence, AA_TO_ID
from utils import MSASampler, Uniclust30_Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_USE_CUDA_DSA'] = "0"

_checkpoints = {"protmamba_foundation": "/nvme1/common/mamba_100M_FIM_checkpoint_32k-100000",
                "protmamba_finetuned": "/nvme1/common/mamba_100M_FIM_finetuned_32k_checkpoint-17000",

                }

val_names = ['CBS_HUMAN_Sun_2020',
             'R1AB_SARS2_Flynn_2022',
             'CAR11_HUMAN_Meitlis_2020_lof',
             'HXK4_HUMAN_Gersing_2023_abundance',
             'PPM1D_HUMAN_Miller_2022',
             'CUE1_YEAST_Tsuboyama_2023_2MYX',
             'HIS7_YEAST_Pokusaeva_2019',
             'PITX2_HUMAN_Tsuboyama_2023_2L7M',
             'DYR_ECOLI_Nguyen_2023',
             'SCN5A_HUMAN_Glazer_2019',
             'RDRP_I33A0_Li_2023',
             'SHOC2_HUMAN_Kwon_2022',
             'KKA2_KLEPN_Melnikov_2014',
             'VILI_CHICK_Tsuboyama_2023_1YU5',
             'GDIA_HUMAN_Silverstein_2021',
             'AMFR_HUMAN_Tsuboyama_2023_4G3O',
             'KCNE1_HUMAN_Muhammad_2023_expression',
             'A0A2Z5U3Z0_9INFA_Wu_2014',
             'TRPC_SACS2_Chan_2017',
             'S22A1_HUMAN_Yee_2023_abundance']

def single_mutational_landscape(model, msa, num_sequences=50, mutated_pos=None, batch_size=64, ):
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
        target_full = target.clone()
        tokens = tokenizer(context_msa, concatenate=True)
        batch_mut_effects = []
        batch_logits = []
        for j in range(i, min(i + batch_size, sequence_length)):
            if j not in mutated_pos:
                batch_mut_effects.append(torch.zeros(20).cpu())
                continue
            gt = target_full[0, j + 1].int().item()
            use_fim = {"<mask-1>": ((j + 1, j + 2), 1)}
            """if sequence_length > 1500:
                target = tokenizer([target_sequence[0][max(0,j-750):j+750]], concatenate=True)
                tokens = tokenizer([seq[max(0,j-750):j+750] for seq in context_msa], concatenate=True)
                if j>750:
                    use_fim = {f"<mask-1>": ((750, 751), 1)}"""
            input_seq, targ_pos, is_fim_dict = prepare_target(target, use_fim=use_fim)
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
            logits = torch.tensor(output["scores"])
            logits_mut = logits[0, 0].log_softmax(-1)
            mut_effect = logits_mut - logits_mut[gt]
            mut_effect = mut_effect[4:24]
            batch_mut_effects.append(torch.tensor(mut_effect).cpu())
            batch_logits.append(torch.tensor(logits_mut).cpu())
        all_mut_effects.extend(batch_mut_effects)
        all_logits.extend(batch_logits)
    all_mut_effects = torch.stack(all_mut_effects, 0)
    all_logits = torch.stack(all_logits, 0)
    return all_mut_effects, all_logits


def multiple_mutational_landscape_iid(landscape, mutation_dict):
    all_mut_effects = {}
    for n_mut, mutation_list in mutation_dict.items():
        all_mut_effects[n_mut] = []
        mut_effect = torch.zeros(len(mutation_list))
        for i in tqdm(range(0, len(mutation_list))):
            muts = mutation_list[i]
            for j in range(n_mut):
                mut_effect[i] += landscape[muts[j][0], AA_TO_ID[muts[j][1]] - 4]
        all_mut_effects[n_mut] = mut_effect.cpu()

    return torch.cat([torch.tensor(all_mut_effects[n_mut]) for n_mut in all_mut_effects.keys()], 0).cpu()


def ensemble_landscapes(folders,
                        save_path,
                        overwrite=True,
                        validation_only=False,
                        weights=None):
    r"""Ensemble the landscapes from different folders

    Args:
        folders (list): list of folders containing the landscapes
        save_path (str): path to save the ensembled landscapes
        overwrite (bool): whether to overwrite the existing files
        validation_only (bool): whether to evaluate only the validation set
        weights (dict): weights for each folder
    """
    if not os.path.exists(f"{save_path}"):
        os.mkdir(f"{save_path}")
    res = {}
    for i, row in database_df.iterrows():
        name = row["DMS_id"]
        if validation_only and name not in val_names:
            continue
        if f"{name}_landscape.pt" in os.listdir(save_path) and not overwrite:
            landscape, gt_landscape = torch.load(f"{save_path}/{name}_landscape.pt")
        else:
            landscape, gt_landscape = None, None
            for folder in folders:
                if f"{name}_landscape.pt" not in os.listdir(folder):
                    continue
                landscape_, gt_landscape = torch.load(f"{folder}/{name}_landscape.pt")
                if weights is not None:
                    if landscape is None:
                        landscape = weights[folder] * landscape_
                    else:
                        landscape += weights[folder] * landscape_
                else:
                    if landscape is None:
                        landscape = landscape_
                    else:
                        landscape += landscape_
            if landscape is None:
                continue
            torch.save((landscape, gt_landscape), f"{save_path}/{name}_landscape.pt")

        sp = spearmanr(gt_landscape, landscape)[0]
        print("Dataset:", name, "Spearman:", sp)
        res[name] = sp

        torch.save(res, f"{save_path}/spearman.pt")


def add_retrievals(alpha,
                   save_path,
                   csv_folder,
                   msa_folder,
                   out_folder,
                   database_df,
                   cache_dir,
                   validation_only=True,
                   msa_type="colabfold",
                   ):
    r"""
    Add the retrievals to the landscapes

    Args:
        alpha (float): alpha for the retrieval
        save_path (str): path to the output folder
        csv_folder (str): path to the csv folder
        msa_folder (str): path to the msa folder
        out_folder (str): path to the output folder
        database_df (pd.DataFrame): dataframe containing the database
        cache_dir (str): path to the cache directory
        validation_only (bool): whether to evaluate only the validation set
        msa_type (str): type of MSA
    """
    if not os.path.exists(f"{save_path}"):
        os.mkdir(f"{save_path}")

    csv_files = os.listdir(csv_folder)
    csv_files = [f for f in csv_files if f.endswith(".csv")]
    csv_files_to_prot = {}
    for filename in csv_files:
        prot = "_".join(filename.split("_")[:2])
        csv_files_to_prot[filename] = prot

    res = {}
    for i, row in list(database_df.iterrows())[::-1]:

        name = row["DMS_id"]
        if validation_only and name not in val_names:
            continue
        csv_filename = row["DMS_filename"]
        msa_filename = csv_files_to_prot[csv_filename]
        msa_start = row["MSA_start"] - 1
        print("Processing:", name)
        torch.cuda.empty_cache()
        if "indels" in csv_filename:
            continue
        if os.path.exists(msa_folder + msa_filename + ".a3m", ):
            msa = load_from_file(msa_folder + msa_filename + ".a3m", )
        elif os.path.exists(msa_folder + msa_filename.split("_")[0] + ".a3m", ):
            msa = load_from_file(msa_folder + msa_filename.split("_")[0] + ".a3m", )
        elif os.path.exists(msa_folder + msa_filename + ".a2m", ):
            msa = load_from_file(msa_folder + msa_filename + ".a2m", )

        else:
            print("No MSA found for:", msa_folder + msa_filename + ".a3m", )
            continue
        if f"{name}_landscape.pt" in os.listdir(save_path):
            landscape, gt_mut_landscape = torch.load(f"{save_path}/{name}_landscape.pt")
        else:
            msa = [seq for seq in msa]
            msa_tokens = [[AA_TO_ID[aa.upper()] for aa in seq] for seq in msa]
            msa_tokens = np.array(msa_tokens)
            one_hot_tokens = np.zeros((len(msa_tokens), len(msa_tokens[0]), 40))
            one_hot_tokens[np.arange(len(msa_tokens))[:, None], np.arange(len(msa_tokens[0])), msa_tokens] = 1
            print("Processing:", name)
            if os.path.exists(os.path.join(cache_dir, name + f"_colabfold.npy")):
                weights = np.load(os.path.join(cache_dir, name + f"_colabfold.npy"))
            else:
                sampler = MSASampler(0.98, 0.7)
                weights = sampler.get_weights(msa_tokens)[1]
                np.save(os.path.join(cache_dir, name + f"_colabfold.npy"),
                        weights) if msa_type == "colabfold" else np.save(os.path.join(cache_dir, name + f".npy"),
                                                                         weights)
                weights = np.load(
                    os.path.join(cache_dir, name + f"_colabfold.npy")) if msa_type == "colabfold" else np.load(
                    os.path.join(cache_dir, name + f".npy"))
            one_hot_tokens = one_hot_tokens * weights[:, None, None]
            one_hot_tokens = one_hot_tokens.sum(0)
            one_hot_tokens = one_hot_tokens[:, 4:24] + 1 / len(msa)
            one_hot_tokens_sum = one_hot_tokens.sum(-1)
            one_hot_tokens = one_hot_tokens / one_hot_tokens_sum[:, None]

            one_hot_tokens = torch.tensor(one_hot_tokens).float()

            df = pd.read_csv(csv_folder + csv_filename)
            gt_mut_landscape_single = torch.ones((len(msa[0]), 20)) * np.inf
            gt_mut_lanscape_multiple_dict = {}
            multiple_mutation_dict = {}
            single_mutation_only = True
            mutated_pos = set()
            for i, row in df.iterrows():
                muts = row["mutant"].split(":")
                if len(muts) == 1:
                    mut_pos = int(row["mutant"][1:-1]) - msa_start - 1
                    mut_aa = row["mutant"][-1]
                    eff = float(row["DMS_score"])
                    mut_aa_id = AA_TO_ID[mut_aa] - 4
                    if mut_pos > len(msa[0]) - 1:
                        continue
                    gt_mut_landscape_single[mut_pos, mut_aa_id] = eff
                    mutated_pos.add(mut_pos)
                else:
                    single_mutation_only = False
                    muts = [(int(mut[1:-1]) - msa_start - 1, mut[-1]) for mut in muts]
                    n_mut = len(muts)
                    if n_mut not in multiple_mutation_dict:
                        multiple_mutation_dict[n_mut] = []
                        gt_mut_lanscape_multiple_dict[n_mut] = []
                    multiple_mutation_dict[n_mut].append(muts)
                    gt_mut_lanscape_multiple_dict[n_mut].append(float(row["DMS_score"]))
                    for mut in muts:
                        mutated_pos.add(mut[0])
            keep_idx = torch.where(gt_mut_landscape_single != np.inf)

            gt = msa_tokens[0]
            logits = one_hot_tokens.log()
            logits = logits - logits[torch.arange(len(logits)), gt - 4][:, None]

            single_mutational_logits_retrieval = logits[keep_idx]
            if not single_mutation_only:
                multiple_mutational_landscape_retrieval = multiple_mutational_landscape_iid(logits,
                                                                                            multiple_mutation_dict)
                landscape_retrieval = torch.cat(
                    [single_mutational_logits_retrieval, multiple_mutational_landscape_retrieval], 0)
            else:
                landscape_retrieval = single_mutational_logits_retrieval
            try:
                landscape_mamba, gt_mut_landscape = torch.load(f"{out_folder}/{name}_landscape.pt")
            except:
                continue
            landscape_mamba = landscape_mamba / 3
            landscape = alpha * landscape_retrieval + (1 - alpha) * landscape_mamba

        sp = spearmanr(gt_mut_landscape, landscape)[0]
        print("Dataset:", name, "Spearman:", sp)
        res[name] = sp
        torch.save((landscape, gt_mut_landscape), f"{save_path}/{name}_landscape.pt")
    torch.save(res, f"{save_path}/spearman.pt")


def evaluate_all_landscapes(checkpoint,
                            prefix,
                            n_tokens_,
                            csv_folder,
                            msa_folder,
                            out_folder,
                            database_df,
                            cache_dir,
                            overwrite=True,
                            validation_only=True,
                            max_similarity=0.98,
                            max_dissimilarity=0.7,
                            max_msa_length=50,
                            ensembling=3, ):
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
        overwrite (bool): whether to overwrite the existing files
        validation_only (bool): whether to evaluate only the validation set
        max_similarity (float): maximum similarity for PoET sampling
        max_dissimilarity (float): maximum dissimilarity for PoET sampling
    """
    save_path = f"{out_folder}/{prefix}_{n_tokens_}/" if n_tokens_ is not None else f"{out_folder}/{prefix}_msalength_{max_msa_length}/"
    if not os.path.exists(f"{save_path}"):
        os.mkdir(f"{save_path}")

    csv_files = os.listdir(csv_folder)
    csv_files = [f for f in csv_files if f.endswith(".csv")]
    csv_files_to_prot = {}
    for filename in csv_files:
        prot = "_".join(filename.split("_")[:2])
        csv_files_to_prot[filename] = prot

    res = {}
    for i, row in list(database_df.iterrows())[::-1]:

        name = row["DMS_id"]
        if validation_only and name not in val_names:
            continue
        model = load_model(checkpoint,
                           model_class=MambaLMHeadModelwithPosids,
                           device="cuda",
                           dtype=torch.bfloat16,
                           checkpoint_mixer="ckpmix" in prefix).eval()

        csv_filename = row["DMS_filename"]
        msa_filename = csv_files_to_prot[csv_filename]
        msa_start = row["MSA_start"] - 1
        msa_len = row["MSA_len"]
        print("Processing:", name)
        torch.cuda.empty_cache()
        if "indels" in csv_filename:
            continue
        if os.path.exists(msa_folder + msa_filename + ".a3m", ):
            msa = load_from_file(msa_folder + msa_filename + ".a3m", )
        elif os.path.exists(msa_folder + msa_filename.split("_")[0] + ".a3m", ):
            msa = load_from_file(msa_folder + msa_filename.split("_")[0] + ".a3m", )
        else:
            print("No MSA found for:", msa_folder + msa_filename + ".a3m", )
            continue
        msa = [seq[msa_start:msa_start + msa_len] for seq in msa]
        first_seq = msa[0]
        if n_tokens_ is not None:
            len_first_seq = min(len(first_seq), 1500)
            msa_length = n_tokens_ // len_first_seq
        else:
            n_tokens = len(first_seq) * max_msa_length
            n_tokens = min(n_tokens, 200000)
            msa_length = n_tokens // len(first_seq)
        msa_length = min(msa_length, 800)

        print("Processing:", name, " with length:", msa_length)
        sampler = MSASampler(max_similarity, max_dissimilarity)

        df = pd.read_csv(csv_folder + csv_filename)
        gt_mut_landscape_single = torch.ones((len(msa[0]), 20)) * np.inf
        gt_mut_lanscape_multiple_dict = {}
        multiple_mutation_dict = {}
        single_mutation_only = True
        mutated_pos = set()
        for i, row in df.iterrows():
            muts = row["mutant"].split(":")
            if len(muts) == 1:
                mut_pos = int(row["mutant"][1:-1]) - msa_start - 1
                mut_aa = row["mutant"][-1]
                eff = float(row["DMS_score"])
                mut_aa_id = AA_TO_ID[mut_aa] - 4
                if mut_pos > len(msa[0]) - 1:
                    continue
                gt_mut_landscape_single[mut_pos, mut_aa_id] = eff
                mutated_pos.add(mut_pos)
            else:
                single_mutation_only = False
                muts = [(int(mut[1:-1]) - msa_start - 1, mut[-1]) for mut in muts]
                n_mut = len(muts)
                if n_mut not in multiple_mutation_dict:
                    multiple_mutation_dict[n_mut] = []
                    gt_mut_lanscape_multiple_dict[n_mut] = []
                multiple_mutation_dict[n_mut].append(muts)
                gt_mut_lanscape_multiple_dict[n_mut].append(float(row["DMS_score"]))
                for mut in muts:
                    mutated_pos.add(mut[0])
        keep_idx = torch.where(gt_mut_landscape_single != np.inf)
        if f"{name}_landscape.pt" in os.listdir(save_path) and not overwrite:
            landscape, gt_landscape = torch.load(f"{save_path}/{name}_landscape.pt")
        else:
            logits_single = None
            landscape_single = None
            landscape_multiple = None
            for i in range(ensembling):
                sample_idxs_small = sampler.get_sample_idxs(msa=msa, name=name + "_colabfold", size=msa_length,
                                                            result_cache_dir=cache_dir)[1:]
                small_msa = [first_seq] + [first_seq] + [msa[i] for i in sample_idxs_small]
                landscape_single_, logits_single_ = single_mutational_landscape(model, small_msa, len(small_msa) - 1,
                                                                                mutated_pos)
                if i == 0:
                    landscape_single = landscape_single_
                    logits_single = logits_single_
                    if not single_mutation_only:
                        landscape_multiple_ = multiple_mutational_landscape_iid(landscape_single_,
                                                                                multiple_mutation_dict)
                        landscape_multiple = landscape_multiple_
                else:
                    landscape_single += landscape_single_
                    logits_single += logits_single_
                    if not single_mutation_only:
                        landscape_multiple_ = multiple_mutational_landscape_iid(landscape_single_,
                                                                                multiple_mutation_dict)
                        landscape_multiple += landscape_multiple_
            gt_mut_landscape_single_keep = gt_mut_landscape_single[keep_idx]
            landscape_single = landscape_single[keep_idx]
            if not single_mutation_only:
                gt_mut_lanscape_multiple = torch.cat(
                    [torch.tensor(gt_mut_lanscape_) for gt_mut_lanscape_ in gt_mut_lanscape_multiple_dict.values()], 0)
                landscape = torch.cat([landscape_single, landscape_multiple], 0)
                gt_landscape = torch.cat([gt_mut_landscape_single_keep, gt_mut_lanscape_multiple], 0)
            else:
                landscape = landscape_single
                gt_landscape = gt_mut_landscape_single_keep
            torch.save((landscape, gt_landscape), f"{save_path}/{name}_landscape.pt")
            torch.save(logits_single, f"{save_path}/{name}_logits.pt")

        sp = spearmanr(gt_landscape, landscape)[0]
        print("Dataset:", name, "Spearman:", sp, "Single mutation only:", single_mutation_only)
        res[name] = sp
    torch.save(res, f"{save_path}/spearman.pt")


def spearmans(save_path, validation_only=False):
    res = {}
    for i, row in list(database_df.iterrows())[::-1]:
        name = row["DMS_id"]
        if validation_only and name not in val_names:
            continue
        if f"{name}_landscape.pt" in os.listdir(save_path):
            try:
                landscape, gt_landscape = torch.load(f"{save_path}/{name}_landscape.pt")
            except:
                print("error: ", name)
                continue
        else:
            continue
        sp = spearmanr(gt_landscape, landscape)[0]
        print("Dataset:", name, "Spearman:", sp)
        res[name] = sp
    torch.save(res, f"{save_path}/spearman.pt")


def spearmanrs_by_depth(csv_folder, save_path):
    if not os.path.exists(f"{save_path}"):
        os.mkdir(f"{save_path}")

    csv_files = os.listdir(csv_folder)
    csv_files = [f for f in csv_files if f.endswith(".csv")]
    csv_files_to_prot = {}
    for filename in csv_files:
        prot = "_".join(filename.split("_")[:2])
        csv_files_to_prot[filename] = prot

    res = {}
    counter = 0
    all_spearmanrs_by_depth = {}
    for i, row in list(database_df.iterrows())[::-1]:

        name = row["DMS_id"]

        csv_filename = row["DMS_filename"]
        df = pd.read_csv(csv_folder + csv_filename)
        mutations_depth_count = [0, 0, 0, 0, 0, 0]
        for i, row in df.iterrows():
            muts = row["mutant"].split(":")
            n_muts = min(len(muts), 5)
            mutations_depth_count[n_muts] += 1
        cumsums = np.cumsum(mutations_depth_count)
        landscape, gt_landscape = torch.load(f"{save_path}/{name}_landscape.pt")
        spearmanrs_by_depth = []
        for i in range(5):
            if mutations_depth_count[i + 1] > 0:
                spearmanrs_by_depth.append(
                    spearmanr(gt_landscape[cumsums[i]:cumsums[i + 1]], landscape[cumsums[i]:cumsums[i + 1]])[0])
            else:
                spearmanrs_by_depth.append(None)

        spearmanrs_by_depth.append(spearmanr(gt_landscape[cumsums[1], :], landscape[cumsums[1], :])[0])
        all_spearmanrs_by_depth[name] = spearmanrs_by_depth
    torch.save(all_spearmanrs_by_depth, f"{save_path}/all_spearmanrs_by_depth.pt")


def auc(save_path, validation_only=False):
    res = {}
    for i, row in list(database_df.iterrows())[::-1]:
        name = row["DMS_id"]
        cutoff = row["DMS_binarization_cutoff"]

        if validation_only and name not in val_names:
            continue
        if f"{name}_landscape.pt" in os.listdir(save_path):
            try:
                landscape, gt_landscape = torch.load(f"{save_path}/{name}_landscape.pt")
            except:
                print("error: ", name)
                continue
        else:
            continue
        try:
            auc = roc_auc_score(gt_landscape > cutoff, landscape)
        except:
            auc = 0.5
        print("Dataset:", name, "AUC:", auc)
        res[name] = auc
    torch.save(res, f"{save_path}/auc.pt")


def mcc(save_path, validation_only=False):
    # compute mattews correlation coefficient sklearn.metrics.matthews_corrcoef
    res = {}
    for i, row in list(database_df.iterrows())[::-1]:
        name = row["DMS_id"]
        cutoff = row["DMS_binarization_cutoff"]

        if validation_only and name not in val_names:
            continue
        if f"{name}_landscape.pt" in os.listdir(save_path):
            try:
                landscape, gt_landscape = torch.load(f"{save_path}/{name}_landscape.pt")
            except:
                print("error: ", name)
                continue
        else:
            continue
        try:
            mcc = matthews_corrcoef(gt_landscape > cutoff, landscape)
        except:
            mcc = 0
        print("Dataset:", name, "MCC:", mcc)
        res[name] = mcc
    torch.save(res, f"{save_path}/mcc.pt")


def top10percentrecall(save_path, validation_only=False):
    res = {}
    for i, row in list(database_df.iterrows())[::-1]:
        name = row["DMS_id"]
        cutoff = row["DMS_binarization_cutoff"]
        if validation_only and name not in val_names:
            continue
        if f"{name}_landscape.pt" in os.listdir(save_path):
            try:
                landscape, gt_landscape = torch.load(f"{save_path}/{name}_landscape.pt")
            except:
                print("error: ", name)
                continue
        else:
            continue
        top_true = (gt_landscape >= np.percentile(gt_landscape, 90))
        top_model = (landscape >= np.percentile(landscape, 90))

        TP = (top_true) & (top_model)
        recall = TP.sum() / (top_true.sum())
        recall = recall.item()
        print("Dataset:", name, "Top 10% Recall:", recall)
        res[name] = recall
    torch.save(res, f"{save_path}/top10percentrecall.pt")


def grid_search(checkpoints, csv_folder, msa_folder, out_folder, database_df, cache_dir):
    r""" Grid search across different hyperparameters

    Args:
        csv_folder (str): path to the csv folder
        msa_folder (str): path to the msa folder
        out_folder (str): path to the output folder
        database_df (pd.DataFrame): dataframe containing the database
        cache_dir (str): path to the cache directory
    """
    # grid search across:
    dissimalirities = [.7]
    similarities = [.98]
    shuffle = [True]
    possible_n_tokens = [200]
    ensembling = [3]

    for (preprefix, checkpoint), n_tokens, do_shuffle, max_similarity, max_dissimilarity, n_ensemble in product(
            checkpoints, possible_n_tokens, shuffle, similarities, dissimalirities, ensembling):
        prefix = f"{preprefix}_colabfold_msa_poetsampling_maxsim{int(max_similarity * 100)}"
        evaluate_all_landscapes(checkpoint,
                                prefix,
                                None,
                                csv_folder,
                                msa_folder,
                                out_folder,
                                database_df,
                                cache_dir,
                                overwrite=False,
                                validation_only=False,
                                max_similarity=max_similarity,
                                max_dissimilarity=max_dissimilarity,
                                max_msa_length=n_tokens,
                                ensembling=n_ensemble)


if __name__ == "__main__":
    data_dir = "/data2/malbrank/protein_gym"
    csv_folder = f"{data_dir}/substitutions/DMS_ProteinGym_substitutions/"
    out_folder = f"{data_dir}/protein_gym/mut_effects_test/"
    database_df = pd.read_csv(f"{data_dir}/substitutions/DMS_substitutions.csv")
    msa_folder_colabfold = f"{data_dir}/msa/colabfold/"
    msa_folder_for_retrievals = f"{data_dir}/msa/proteingym/"
    cache_folder = f"{data_dir}/cache"
    expe_name = "protmamba_long_finetuned"
    checkpoints = [("new_mamba_ckpmix_ft", "/nvme1/common/mamba_100M_FIM-finetuned_131k_checkpoint-3200"), ]

    # Usage evaluate_all_landscapes
    #evaluate_all_landscapes(checkpoints[0][0], expe_name,None, csv_folder, msa_folder_colabfold, out_folder, database_df, cache_folder, overwrite=False, validation_only=False,)

    # Usage for grid search
    # grid_search(checkpoints, csv_folder, msa_folder_colabfold, out_folder, database_df, cache_folder)

    # Usage for retrievals
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        out_folder_ = f"{out_folder}/protmamba_long_finetuned_200"
        save_path = out_folder_ + f"_with_retrievals_proteingym_{alpha}"
        # add_retrievals(alpha, save_path, csv_folder, msa_folder_for_retrievals, out_folder_, database_df, cache_folder, validation_only=False, msa_type="proteingym")

    # usage for spearamnrs, auc, mcc, top10percentrecall, spearmanrs_by_depth
    out_folder_ = f"{out_folder}/{expe_name}"
    # spearmans(out_folder_, validation_only=False)
    # auc(out_folder_, validation_only=False)
    # mcc(out_folder_, validation_only=False)
    # top10percentrecall(out_folder_, validation_only=False)
    # spearmanrs_by_depth(csv_folder, out_folder_)

    # Usage for ensembling
    to_ensemble_folders = [
        out_folder + "protmamba_long_finetuned_25",
        out_folder + "protmamba_long_finetuned_50",
        out_folder + "protmamba_long_finetuned_100",
        out_folder + "protmamba_long_finetuned_200",
        out_folder + "protmamba_long_finetuned_400",
    ]
    out_ensembled_folder = out_folder + "protmamba_long_finetuned_25_to_400"
    # ensemble_landscapes(to_ensemble_folders, out_ensembled_folder, validation_only=False)

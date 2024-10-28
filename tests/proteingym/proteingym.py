import os, sys

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mamba_ssm.utils.generation import *

from ProtMamba_ssm import MambaLMHeadModelwithPosids, load_model, load_from_file, tokenizer, AA_TO_ID
from tests.utils import MSASampler

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

CACHE_DIR = "/data2/malbrank/protein_gym/cache"
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)


def prepare_target_for_mut_landscape(target, fim=None):
    """Prepare the target for the mutational landscape generation.

    Args:
        target (torch.Tensor): the target sequence to prepare.
        fim (list): a list of tuples indicating the start and end of the FIM regions."""
    if fim is None:
        return target, torch.arange(0, target.size(1), dtype=torch.long), 0
    target_seq = target[0].clone()
    target_pos_ids = torch.arange(0, target_seq.size(0), dtype=torch.long).clamp(0, 1500)
    in_fim_target = []
    not_in_fim_target = []
    in_fim_target_pos_ids = []
    not_in_fim_target_pos_ids = []
    cursor = 0
    for idx, (i, j) in enumerate(fim):
        not_in_fim_target.append(target_seq[cursor: i])
        not_in_fim_target.append(torch.ones(1, dtype=torch.long) * AA_TO_ID[f"<mask-{idx + 1}>"])
        not_in_fim_target_pos_ids.append(target_pos_ids[cursor: i])
        not_in_fim_target_pos_ids.append(target_pos_ids[i:i + 1])

        in_fim_target.append(torch.ones(1, dtype=torch.long) * AA_TO_ID[f"<mask-{idx + 1}>"])
        in_fim_target.append(target_seq[i: j])
        in_fim_target_pos_ids.append(target_pos_ids[i:i + 1])
        in_fim_target_pos_ids.append(target_pos_ids[i: j])
        cursor = j
    not_in_fim_target.append(target_seq[cursor:])
    not_in_fim_target.append(torch.ones(1, dtype=torch.long) * 2)
    not_in_fim_target = torch.cat(not_in_fim_target, 0)

    not_in_fim_target_pos_ids.append(target_pos_ids[cursor:])
    not_in_fim_target_pos_ids.append(torch.zeros(1, dtype=torch.long))
    not_in_fim_target_pos_ids = torch.cat(not_in_fim_target_pos_ids, 0)

    in_fim_target = torch.cat(in_fim_target, 0)
    in_fim_target_pos_ids = torch.cat(in_fim_target_pos_ids, 0)
    return (torch.cat([not_in_fim_target, in_fim_target], 0)[None],
            torch.cat([not_in_fim_target_pos_ids, in_fim_target_pos_ids], 0)[None],
            len(not_in_fim_target))


def mutational_landscape_full(model, mutations_lists, msa, batch_size=16, device="cuda:0"):
    """Generate the mutational landscape of a single protein sequence by having full probabiltiies when possible.
    Args:
        model (MambaLMHeadModelwithPosids): the model to use for generation.
        mutations_lists (list): a list of lists of mutations to evaluate.
        msa (list): a list of sequences in the MSA.
        batch_size (int): the batch size to use for generation.
        """
    target_sequence = msa[:1]
    context_msa = msa[::-1]
    all_mut_effects = []
    target: torch.Tensor = tokenizer(target_sequence, concatenate=True)
    context_tokens = tokenizer(context_msa, concatenate=False)
    context_pos_ids = torch.cat([torch.arange(0, len(seq), dtype=torch.long) for seq in context_tokens], 0)[None]
    context_tokens = torch.cat(context_tokens, 0)[None]

    targets = []
    gts = []
    start_fims = []
    mut_pointer = {}
    for i, mut_list in enumerate(mutations_lists):
        gt = target.clone()
        tg = target.clone()
        fim_list = []
        sorted(mut_list, key=lambda x: x[0])
        for mut in mut_list:
            if len(mut) > 5:
                return mutational_landscape(model, mutations_lists, msa, batch_size=batch_size, device=device)
            tg[0, mut[0] + 1] = AA_TO_ID[mut[1]]
            fim_list.append([mut[0] + 1, mut[0] + 2])
            if str(mut_list[:-1]) + "/" + str(mut_list[-1][0]) not in mut_pointer:
                mut_pointer[str(mut_list[:-1]) + "/" + str(mut_list[-1][0])] = []
            mut_pointer[str(mut_list[:-1]) + "/" + str(mut_list[-1][0])].append(i)
        target_seq, target_pos_ids, start_fim = prepare_target_for_mut_landscape(tg, fim=fim_list)
        gt_seq, gt_pos_ids, gt_start_fim = prepare_target_for_mut_landscape(gt, fim=fim_list)
        targets.append((target_seq, target_pos_ids))
        gts.append((gt_seq, gt_pos_ids))
        start_fims.append(start_fim)
    reverse_single_mut_pointer = {}
    for k, v in mut_pointer.items():
        for vv in v:
            reverse_single_mut_pointer[vv] = v[0]
    non_redundant = [v[0] for v in mut_pointer.values()]
    non_redundant_vectors_logits = {}
    non_redundant_gt_vectors_logits = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(non_redundant), batch_size), desc="Generating landscapes"):
            inputs_ids = []
            input_pos_ids = []
            gt_inputs_ids = []
            gt_input_pos_ids = []
            seq_length = []
            target_seqs = []
            gt_seqs = []
            for j in range(i, min(i + batch_size, len(non_redundant))):
                cursor = non_redundant[j]
                target_seq, target_pos_ids = targets[cursor]
                target_seqs.append(target_seq)
                gt_seq, gt_pos_ids = gts[cursor]
                gt_seqs.append(gt_seqs)

                inputs_ids.append(torch.cat([context_tokens, target_seq], 1))
                input_pos_ids.append(torch.cat([context_pos_ids, target_pos_ids], 1))
                gt_inputs_ids.append(torch.cat([context_tokens, gt_seq], 1))
                gt_input_pos_ids.append(torch.cat([context_pos_ids, gt_pos_ids], 1))
                seq_length.append(target_seq.size(1) + context_tokens.size(1))
            max_seq_len = max(seq_length)
            inputs_ids = torch.cat([torch.cat([x, torch.zeros(1, max_seq_len - x.size(1), dtype=torch.long)],
                                              1) if max_seq_len - x.size(1) > 0 else x for x in inputs_ids], 0)
            input_pos_ids = torch.cat([torch.cat([x, torch.zeros(1, max_seq_len - x.size(1), dtype=torch.long)],
                                                 1) if max_seq_len - x.size(1) > 0 else x for x in input_pos_ids], 0)
            gt_inputs_ids = torch.cat([torch.cat([x, torch.zeros(1, max_seq_len - x.size(1), dtype=torch.long)],
                                                 1) if max_seq_len - x.size(1) > 0 else x for x in gt_inputs_ids], 0)
            gt_input_pos_ids = torch.cat([torch.cat([x, torch.zeros(1, max_seq_len - x.size(1), dtype=torch.long)],
                                                    1) if max_seq_len - x.size(1) > 0 else x for x in gt_input_pos_ids],
                                         0)

            target_output = model(inputs_ids.to(device), position_ids=input_pos_ids.to(device), ).logits.cpu().float()
            gt_output = model(gt_inputs_ids.to(device), position_ids=gt_input_pos_ids.to(device), ).logits.cpu().float()

            for j in range(len(seq_length)):
                start_fim = start_fims[j] + context_tokens.size(1)
                seq_len = seq_length[j] + context_tokens.size(1)
                target_logits = target_output[j, start_fim - 1:seq_len - 1].log_softmax(-1)
                gt_logits = gt_output[j, start_fim - 1:seq_len - 1].log_softmax(-1)
                target_logits = target_logits[1::2]
                gt_logits = gt_logits[1::2]
                non_redundant_vectors_logits[non_redundant[i + j]] = target_logits
                non_redundant_gt_vectors_logits[non_redundant[i + j]] = gt_logits
    for i in range(0, len(mutations_lists)):
        target_logits = non_redundant_vectors_logits[reverse_single_mut_pointer[i]]
        gt_logits = non_redundant_gt_vectors_logits[reverse_single_mut_pointer[i]]
        target_seq = targets[i][0]
        start_fim = start_fims[i]
        gt_seq = gts[i][0]

        mut_effect = (target_logits[:, target_seq[0, start_fim:][1::2]] - gt_logits[:,
                                                                          gt_seq[0, start_fim:][1::2]]).sum().item()
        all_mut_effects.append(mut_effect)
    return all_mut_effects


def mutational_landscape(model, mutations_lists, msa, batch_size=8, device="cuda:0"):
    """Generate the mutational landscape of a single protein sequence independantly summing pointwise effects.
    Args:
        model (MambaLMHeadModelwithPosids): the model to use for generation.
        mutations_lists (list): a list of lists of mutations to evaluate.
        msa (list): a list of sequences in the MSA.
        batch_size (int): the batch size to use for generation.
        device (str): the device to use for generation.
        """
    target_sequence = msa[:1]
    context_msa = msa[::-1]
    all_mut_effects = []
    target: torch.Tensor = tokenizer(target_sequence, concatenate=True)
    context_tokens = tokenizer(context_msa, concatenate=False)
    context_pos_ids = torch.cat([torch.arange(0, len(seq), dtype=torch.long) for seq in context_tokens], 0)[None].clamp(
        0, 2000)
    context_tokens = torch.cat(context_tokens, 0)[None]
    mut_pointer = {}
    for i, mut_list in enumerate(mutations_lists):
        for mut in mut_list:
            if mut[0] not in mut_pointer:
                mut_pointer[mut[0]] = []
            mut_pointer[mut[0]].append(i)
    non_redundant = list(mut_pointer.keys())
    non_redundant_vectors_logits = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(non_redundant), batch_size), desc="Generating landscapes"):
            inputs_ids = []
            input_pos_ids = []
            seq_length = []
            target_seqs = []
            for j in range(i, min(i + batch_size, len(non_redundant))):
                cursor = non_redundant[j]
                target_seq, target_pos_ids, start = prepare_target_for_mut_landscape(target,
                                                                                     fim=[[cursor + 1, cursor + 2]])
                target_seqs.append(target_seq)

                inputs_ids.append(torch.cat([context_tokens, target_seq], 1))
                input_pos_ids.append(torch.cat([context_pos_ids, target_pos_ids], 1))
                seq_length.append(target_seq.size(1) + context_tokens.size(1))
            max_seq_len = max(seq_length)
            inputs_ids = torch.cat([torch.cat([x, torch.zeros(1, max_seq_len - x.size(1), dtype=torch.long)],
                                              1) if max_seq_len - x.size(1) > 0 else x for x in inputs_ids], 0)
            input_pos_ids = torch.cat([torch.cat([x, torch.zeros(1, max_seq_len - x.size(1), dtype=torch.long)],
                                                 1) if max_seq_len - x.size(1) > 0 else x for x in input_pos_ids], 0)

            target_output = model(inputs_ids.to(device), position_ids=input_pos_ids.to(device), ).logits.cpu().float()

            for j in range(len(seq_length)):
                seq_len = seq_length[j]
                target_logits = target_output[j, seq_len - 2][4:24].log_softmax(-1)
                non_redundant_vectors_logits[non_redundant[i + j]] = target_logits
    for i in range(0, len(mutations_lists)):
        logits = [non_redundant_vectors_logits[mut[0]] for mut in mutations_lists[i]]
        target_tokens = [AA_TO_ID[mut[1]] - 4 for mut in mutations_lists[i]]
        gt_tokens = [target[0, mut[0] + 1].item() - 4 for mut in mutations_lists[i]]

        mut_effect = (torch.stack(logits)[torch.arange(len(logits)), target_tokens] - torch.stack(logits)[
            torch.arange(len(logits)), gt_tokens]).sum().item()
        all_mut_effects.append(mut_effect)
    return all_mut_effects


def add_retrievals(alpha,
                   save_path,
                   csv_folder,
                   msa_folder,
                   database_df,
                   cache_dir,
                   validation_only=True,
                   ):
    r"""
    Add the retrievals to the landscapes

    Args:
        alpha (float): alpha for the retrieval
        save_path (str): path to the save folder to save spearman correlation
        csv_folder (str): path to the csv folder that contains the protmamba predictions
        msa_folder (str): path to the msa folder that contains colabfold msa
        database_df (pd.DataFrame): dataframe containing the protein gym list of DMS
        cache_dir (str): path to the cache directory
        validation_only (bool): whether to evaluate only the validation set
    """
    if not os.path.exists(f"{save_path}"):
        os.mkdir(f"{save_path}")
    res = {}
    for i, row in list(database_df.iterrows()):
        name = row["DMS_id"]
        if validation_only and name not in val_names:
            continue
        csv_filename = row["DMS_filename"]
        msa_filename = row["MSA_filename"][:-4]
        msa_start = row["MSA_start"] - 1
        print("Processing:", name)
        torch.cuda.empty_cache()
        if "indels" in csv_filename:
            continue
        if os.path.exists(msa_folder + msa_filename + ".a2m", ):
            msa = load_from_file(msa_folder + msa_filename + ".a2m", )
        else:
            print("No MSA found for:", msa_folder + msa_filename + ".a2m", )
            continue
        df = pd.read_csv(csv_folder + csv_filename)
        if f"retrieved_score_alpha_{alpha}" in df.columns:
            ()
        else:
            msa = [seq for seq in msa]
            msa_tokens = [[AA_TO_ID[aa.upper()] for aa in seq] for seq in msa]
            msa_tokens = np.array(msa_tokens)
            one_hot_tokens = np.zeros((len(msa_tokens), len(msa_tokens[0]), 40))
            one_hot_tokens[np.arange(len(msa_tokens))[:, None], np.arange(len(msa_tokens[0])), msa_tokens] = 1
            print("Processing:", name)
            if os.path.exists(os.path.join(cache_dir, name + f".npy")):
                weights = np.load(os.path.join(cache_dir, name + f".npy"))
            else:
                sampler = MSASampler(0.98, 0.7)
                weights = sampler.get_weights(msa_tokens)[1]
                np.save(os.path.join(cache_dir, name + f".npy"), weights)

            one_hot_tokens = one_hot_tokens * weights[:, None, None]
            one_hot_tokens = one_hot_tokens.sum(0)
            one_hot_tokens = one_hot_tokens[:, 4:24] + 1 / len(msa)
            one_hot_tokens_sum = one_hot_tokens.sum(-1)
            one_hot_tokens = one_hot_tokens / one_hot_tokens_sum[:, None]
            one_hot_tokens = torch.tensor(one_hot_tokens).float()

            gt = msa_tokens[0]
            logits = one_hot_tokens.log()
            logits = logits - logits[torch.arange(len(logits)), gt - 4][:, None]

            df = pd.read_csv(csv_folder + csv_filename)
            retrieved_scores = []
            for i, row in tqdm(df.iterrows()):
                muts = row["mutant"].split(":")
                pred_score = float(row["predicted_score"])
                iid_score = 0
                for mut in muts:
                    pos = int(mut[1:-1]) - msa_start - 1
                    aa = AA_TO_ID[mut[-1]]
                    iid_score += logits[pos, aa - 4].item()
                retrieved_score = (1 - alpha) * pred_score + alpha * iid_score
                retrieved_scores.append(retrieved_score)
            df[f"retrieved_score_alpha_{alpha}"] = retrieved_scores
            df.to_csv(csv_folder + csv_filename, index=False)
        landscape = np.array(df[f"retrieved_score_alpha_{alpha}"])
        gt_mut_landscape = np.array(df["DMS_score"])
        sp = spearmanr(gt_mut_landscape, landscape)[0]
        res[name] = sp
        print("Dataset:", name, "Spearman:", sp)
    torch.save(res, f"{save_path}/spearman.pt")


def evaluate_one_csv(csv_file, out_file, cache_dir, model_config, mamba_config, retrieval_config=None,
                     device="cuda:0"):
    r"""
    Evaluate one landscape

    Args:
        csv_file (str): path to the csv file
        out_file (str): path to the output file
        cache_dir (str): path to the cache directory
        model_config (dict): dictionary containing the model configuration
        mamba_config (dict): dictionary containing the mamba configuration
        retrieval_config (dict): dictionary containing the retrieval configuration
        device (str): device to use for evaluation
    """
    name = mamba_config.get("name", csv_file.split("/")[-1][:-4])
    n_tokens = mamba_config.get("n_tokens", None)
    max_msa_length = mamba_config.get("max_msa_length", 50)
    msa_start = mamba_config.get("msa_start", 0)
    msa_len = mamba_config.get("msa_len", None)
    msa_file = mamba_config.get("msa_file", None)
    overwrite = mamba_config.get("overwrite", True)
    max_similarity = mamba_config.get("max_similarity", 0.98)
    max_dissimilarity = mamba_config.get("max_dissimilarity", 0.7)
    ensembling = mamba_config.get("ensembling", 1)
    mutational_landscap_fctn = mamba_config.get("mutation_landcape_fct", mutational_landscape)

    if os.path.exists(out_file) and not overwrite:
        df = pd.read_csv(out_file)
        sp = spearmanr(df["DMS_score"], df["predicted_score"])[0]
        print("Dataset:", name, "Spearman:", sp)
        return df, sp

    msa = load_from_file(msa_file)
    msa = [seq[msa_start:] for seq in msa]
    if msa_len is not None:
        msa = [seq[:msa_len] for seq in msa]
    first_seq = msa[0]
    if n_tokens is not None:
        len_first_seq = min(len(first_seq), 1500)
        msa_length = n_tokens // len_first_seq
    else:
        n_tokens = len(first_seq) * max_msa_length
        n_tokens = min(n_tokens, 200000)
        msa_length = n_tokens // len(first_seq)
    msa_length = min(msa_length, 800)
    print("Processing:", name, " with length:", msa_length)

    df = pd.read_csv(csv_file)
    mutations_lists = []
    labels = []
    gt_landscape = []

    for i, row in df.iterrows():
        labels.append(row["mutant"])
        muts = row["mutant"].split(":")
        muts = [(int(mut[1:-1]) - msa_start - 1, mut[-1]) for mut in muts]
        mutations_lists.append(muts)
        gt_landscape.append(float(row["DMS_score"]))
    gt_landscape = np.array(gt_landscape)
    landscapes = []

    model = load_model(model_config.get("checkpoint", None),
                       model_class=model_config.get("model_class", MambaLMHeadModelwithPosids),
                       device=device,
                       dtype=torch.bfloat16,
                       checkpoint_mixer=model_config.get("checkpoint_mixer", False)).eval()
    sampler = MSASampler(max_similarity, max_dissimilarity)
    for i in range(ensembling):
        sample_idxs_small = sampler.get_sample_idxs(msa=msa, name=name + "_colabfold", size=msa_length,
                                                    result_cache_dir=cache_dir)[1:]
        small_msa = [first_seq] + [msa[i] for i in sample_idxs_small]
        landscape = mutational_landscap_fctn(model, mutations_lists, small_msa, device=device)
        landscapes.append(np.array(landscape))
    sum_landscape = np.mean(landscapes, 0)
    sp = spearmanr(gt_landscape, sum_landscape)[0]
    print("Dataset:", name, "Spearman:", sp)
    result_dict = {"mutant": labels,
                   "DMS_score": gt_landscape,
                   "predicted_score": sum_landscape}
    if ensembling > 1:
        for i in range(ensembling):
            result_dict[f"predicted_score_{i}"] = landscapes[i]
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(f"{out_file}/{name}.csv", index=False)
    if retrieval_config is not None:
        alpha = retrieval_config.get("alpha", 0.5)
        msa_folder = retrieval_config.get("msa_folder", "")
        add_retrievals(alpha=alpha,
                       save_path="/".join(out_file.split("/")[:-1])+f"_retrieval_{alpha}/",
                       csv_folder=csv_folder,
                       msa_folder=msa_folder,
                       database_df=database_df,
                       cache_dir=cache_dir,
                       validation_only=False, )
    return result_df, sp


def evaluate_all_landscapes(prefix,
                            csv_folder,
                            msa_folder,
                            out_folder,
                            database_df,
                            model_config,
                            mamba_config,
                            retrieval_config,
                            cache_dir=CACHE_DIR,
                            validation_only=True,
                            device="cuda:0"):
    r"""
    Evaluate all spearmanr correlation for every landscape

    Args:
        prefix (str): prefix for the output folder
        csv_folder (str): path to the csv folder
        msa_folder (str): path to the msa folder
        out_folder (str): path to the output folder
        database_df (pd.DataFrame): dataframe containing the protein gym list of DMS
        mamba_config (dict): dictionary containing the mamba configuration
        retrieval_config (dict): dictionary containing the retrieval configuration
        cache_dir (str): path to the cache directory
        validation_only (bool): whether to evaluate only the validation set
        device (str): device to use for evaluation
    """
    n_tokens_ = mamba_config.get("n_tokens", None)
    max_msa_length = mamba_config.get("max_msa_length", 50)
    overwrite = mamba_config.get("overwrite", True)
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
    for i, row in list(database_df.iterrows()):

        name = row["DMS_id"]
        if validation_only and name not in val_names:
            continue
        if f"{name}_landscape.csv" in os.listdir(save_path) and not overwrite:
            continue
        csv_filename = row["DMS_filename"]
        msa_filename = csv_files_to_prot[csv_filename]
        msa_start = row["MSA_start"] - 1
        msa_len = row["MSA_len"]
        torch.cuda.empty_cache()
        mamba_config.update(
            {"msa_start": msa_start, "msa_len": msa_len, "msa_file": msa_folder + msa_filename + ".a3m"})
        evaluate_one_csv(csv_folder + csv_filename,
                         f"{save_path}/{name}.csv",
                         cache_dir,
                         model_config,
                         mamba_config,
                         retrieval_config,
                         device=device)
    torch.save(res, f"{save_path}/spearman.pt")


def add_other_model(database_df, folder_list, out_folder, column_name_list, weights=None):
    # if weights are not provided, use Linear Regression on val names to combine the landscapes:
    folder_gt = "/data2/malbrank/protein_gym/substitutions/zero_shot_substitutions_scores/Ground_truth/"
    res_csv = [{x["DMS_id"]: pd.read_csv(folder_list[k] + x["DMS_filename"])[column_name_list[k]].values for i, x in
                database_df.iterrows()} for k in range(len(folder_list))]

    res = {}
    X = [np.array([res_csv[i][name] for i in range(len(res_csv))]) for name in database_df["DMS_id"]]
    X = [x.reshape(len(res_csv), -1).T for x in X]

    pred = {x["DMS_id"]: (X[i] * weights).sum(1) for i, x in database_df.iterrows()}
    if not os.path.exists(f"{out_folder}"):
        os.mkdir(f"{out_folder}")
    for i, row in database_df.iterrows():
        name = row["DMS_id"]
        p = pred[name]
        csv_df = pd.read_csv(folder_gt + row["DMS_filename"])
        df = pd.DataFrame({"mutant": csv_df["mutant"].values, "combined": p, "DMS_score": csv_df["DMS_score"].values})
        df.to_csv(f"{out_folder}/{name}.csv", index=False)
        sp = spearmanr(p, df["DMS_score"])[0]
        print("Dataset:", name, "Spearman:", sp)
        res[name] = sp
    torch.save(res, f"{out_folder}/spearman.pt")


if __name__ == "__main__":
    data_dir = "/data2/malbrank/protein_gym"

    model_config = {"checkpoint": "/nvme1/common/mamba_100M_FIM-finetuned_131k_checkpoint-3200",
                    "model_class": MambaLMHeadModelwithPosids,
                    "checkpoint_mixer": False
                    }

    mamba_config = {"n_tokens": None,
                    "max_msa_length": 200,
                    "overwrite": True,
                    "max_similarity": 0.98,
                    "max_dissimilarity": 0.7,
                    "ensembling": 1,
                    "mutation_landcape_fct": mutational_landscape}

    msa_folder_for_retrievals = f"{data_dir}/msa_files/proteingym/"
    retrieval_config = {"alpha": 0.5,
                        "msa_folder": msa_folder_for_retrievals}

    out_folder = f"{data_dir}/mut_effects/"
    database_df = pd.read_csv(f"{data_dir}/substitutions/DMS_substitutions.csv")
    msa_folder_colabfold = f"{data_dir}/msa_files/colabfold/"
    csv_folder = f"{data_dir}/substitutions/DMS_ProteinGym_substitutions/"
    cache_folder = f"{data_dir}/cache"
    expe_name = "protmamba1_finetuned_corrected"

    # Usage evaluate_all_landscape
    evaluate_all_landscapes(expe_name,
                            csv_folder,
                            msa_folder_colabfold,
                            out_folder,
                            database_df,
                            model_config=model_config,
                            mamba_config=mamba_config,
                            retrieval_config=retrieval_config,
                            validation_only=False,
                            device="cuda:0")

    # Usage for retrievals
    for alpha in [0.5]:
        folder = "/data2/malbrank/protein_gym/mut_effects/protmamba1_finetuned_corrected_msalength_200/"
        save_path = out_folder + f"_with_retrievals_proteingym_{alpha}"
        add_retrievals(alpha,
                       save_path,
                       folder,
                       msa_folder_for_retrievals,
                       database_df,
                       cache_folder,
                       validation_only=False,
                       )

__all__ = ['AA_TO_ID', 'MASK_TO_ID', 'ID_TO_AA', 'encode_sequence', 'decode_sequence', 'clean_sequence', 'tokenizer',
           'reorder_masked_sequence', 'load_from_file', 'generate_sequence', 'prepare_dataset_for_fim_generation',
           'prepare_tokens', 'prepare_target', 'load_tensorboard_data', 'filter_datapoints', 'save_to_tensorboard',
           'merge_loggings', 'concatenate_loggings', 'print_number_of_parameters', 'find_fim_indices',
           'compute_metrics']

# Constants
AA_TO_ID = {'<cls>': 0,
            '<pad>': 1,
            '<eos>': 2,
            '<unk>': 3,
            'L': 4,
            'A': 5,
            'G': 6,
            'V': 7,
            'S': 8,
            'E': 9,
            'R': 10,
            'T': 11,
            'I': 12,
            'D': 13,
            'P': 14,
            'K': 15,
            'Q': 16,
            'N': 17,
            'F': 18,
            'Y': 19,
            'M': 20,
            'H': 21,
            'W': 22,
            'C': 23,
            'X': 24,
            'B': 25,
            'U': 26,
            'Z': 27,
            'O': 28,
            '.': 29,
            '-': 30,
            '<null_1>': 31,
            '<mask>': 32}

MASK_TO_ID = {"<mask-1>": 33,
              "<mask-2>": 34,
              "<mask-3>": 35,
              "<mask-4>": 36,
              "<mask-5>": 37,}

AA_TO_ID.update(MASK_TO_ID)

ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}

import numpy as np
import torch
from Bio import SeqIO

# Encoder
def encode_sequence(sequence):
    """Tokenize a sequence of amino acids and add a cls token at the beginning."""
    tokenized_sequence = [AA_TO_ID[aa] if aa in AA_TO_ID else AA_TO_ID['<unk>'] for aa in sequence]
    return [AA_TO_ID['<cls>']] + tokenized_sequence

def decode_sequence(sequence):
    """Decode a sequence of tokens."""
    return "".join([ID_TO_AA[token] if token in ID_TO_AA else "<unk>" for token in sequence])

def clean_sequence(sequence):
    """Remove gaps and convert all residues to upper case."""
    return sequence.replace("-", "").upper()

def tokenizer(sequence_list, concatenate=True):
    """Tokenize a collection of sequences. If the sequences are aligned, the gaps will be removed
    and the insertions (lower case) will be promoted to upper case."""
    # clean and encode all sequences
    sequence_list = [encode_sequence(clean_sequence(sequence)) for sequence in sequence_list]
    if concatenate:
        # concatenate all sequences
        sequences = np.concatenate(sequence_list)
        # convert to tensor and add batch dimension
        return torch.asarray(sequences, dtype=torch.int64)[None,:]
    else:
        return [torch.asarray(sequence, dtype=torch.int64) for sequence in sequence_list]


def reorder_masked_sequence(mask_seq, return_ids=False):
    """
    Reorder a masked sequence to fill the masked positions with the tokens
    that should be there but are positioned after the <eos> token.
    """
    mask_seq = mask_seq.split("<cls>")[0]
    try:
        # Split the sequence and masks
        seq, masks = mask_seq.split("<eos>")
    except:
        return mask_seq
    full_seq = ""
    ids_mask = []
    # Iterate over each mask tag
    for mm in ["<mask-1>", "<mask-2>", "<mask-3>", "<mask-4>", "<mask-5>","<mask-?>"]:
        try:
            # Split the sequence in before and after the mask tag
            seq1, seq2 = seq.split(mm)
            if mm=="<mask-1>":
                # If the mask is the first one, add the sequence before the mask and update the masks
                masks = masks.split("<mask-1>")[1]
                full_seq += seq1
            else:
                # If the mask is not the first one, insert the mask between the two sequence parts
                masks1, masks2 = masks.split(mm)
                ids_mask += [(len(full_seq), len(full_seq)+len(masks1))]
                full_seq += masks1 + seq1
                # Update the masks
                masks = masks2 
            # Update the sequence with the part after the mask
            seq = seq2
        except:
            # If the mask is not found, add the remaining sequence
            ids_mask += [(len(full_seq), len(full_seq)+len(masks))]
            full_seq += masks + seq
            break
    if return_ids:
        return full_seq, ids_mask
    return full_seq

def load_from_file(file_path):
    """Load a collection of sequences from an a3m file."""
    with open(file_path, "r") as f:
        sequences = [str(record.seq) for record in SeqIO.parse(f, "fasta")]
    return sequences

def generate_sequence(model, tokens, position_ids=None, seq_position_ids=None, is_fim=False, max_length=1,
                      temperature=1., top_p=0.0, top_k=1,
                      return_dict_in_generate=False, output_scores=False, teacher_outputs=None,
                      eos_token_id=AA_TO_ID["<cls>"],
                      device="cuda"):
    """Generating, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p. We assume that all sequences in the same batch have the same length.
    """
    input_ids = tokens.to(device)
    position_ids = position_ids.to(device) if position_ids is not None else None
    seq_position_ids = seq_position_ids.to(device) if seq_position_ids is not None else None
    teacher_outputs = teacher_outputs.to(device) if teacher_outputs is not None else None
    # generate sequence
    out = model.generate(input_ids=input_ids,
                         position_ids=position_ids.clamp(0, 2000),
                         seq_position_ids=seq_position_ids,
                         is_fim=is_fim,
                         max_length=max_length,
                         temperature=temperature,
                         top_p=top_p,
                         top_k=top_k,
                         return_dict_in_generate=return_dict_in_generate,
                         output_scores=output_scores,
                         eos_token_id=eos_token_id,
                         teacher_outputs=teacher_outputs)
    sequences = out.sequences
    dic = {"input": [decode_sequence(seq) for seq in sequences[:, :input_ids.shape[-1]].cpu().numpy()],
           "generated": [decode_sequence(seq) for seq in sequences[:, input_ids.shape[-1]:].cpu().numpy()],
           "input_tokens": [seq for seq in sequences[:, :input_ids.shape[-1]].cpu().numpy()],
           "generated_tokens": [seq for seq in sequences[:, input_ids.shape[-1]:].cpu().numpy()]}
    if output_scores:
        dic["scores"] = np.array([el.to(torch.float32).cpu().numpy() for el in out.scores]).transpose(1, 0, 2)
    return dic


def prepare_dataset_for_fim_generation(tokens, pos_ids):
    """
    Function to transform the tokenized training dataset into a format that can be used for FIM generation.
    Splits the input tokens and pos_ids into the FIM part (of the last sequence) and the context part (all
    the previous sequences and the masked part of the last sequence).
    Also returns a dictionary with the positions of the mask tokens in the FIM part.
    """
    def find_mask_positions(tokens_fim):
        """
        Function to find the positions of the mask tokens in the FIM part of the last sequence.
        """
        bool_mask = None
        inds_masks = []
        for ind in MASK_TO_ID.values():
            tmp_bool = tokens_fim[0].cpu().numpy() == ind
            bool_mask = tmp_bool if bool_mask is None else bool_mask | tmp_bool
            inds_masks += [ind]
        return bool_mask, inds_masks
    # find where the FIM part of the last sequence starts
    start_last_fim = np.where(tokens[0].cpu().numpy() == AA_TO_ID["<eos>"])[0][-1]
    start_next_seqs = np.where(tokens[0,start_last_fim+1:].cpu().numpy() == AA_TO_ID["<cls>"])[0]
    end_last_fim = start_last_fim+ 1 +start_next_seqs[0] if len(start_next_seqs) > 0 else tokens.shape[1]
    # split tokens and pos_ids into FIM part and context part
    tokens_to_fim = tokens[:,:start_last_fim+1]
    pos_ids_to_fim = pos_ids[:,:start_last_fim+1]
    tokens_fim = tokens[:,start_last_fim+1:end_last_fim]
    pos_ids_fim = pos_ids[:,start_last_fim+1:end_last_fim]
    # find positions of mask tokens
    bool_mask, inds_masks = find_mask_positions(tokens_fim)
    masked_positions = pos_ids_fim[0,bool_mask]
    mask_dict = {ind: int(pos) for ind, pos in zip(inds_masks, masked_positions)}
    return tokens_to_fim, pos_ids_to_fim, tokens_fim, pos_ids_fim, mask_dict

def prepare_tokens(context_tokens,
                   target_tokens,
                   target_pos_ids,
                   DatasetClass,
                   num_sequences=1,
                   fim_strategy="no-scramble", # "multiple_span"
                   mask_fraction=0.2,
                   max_patches=5,
                   add_position_ids="1d",
                   shuffle=True):
    """Prepare the tokens for the model by applying the FIM strategy and masking the tokens.
    It uses custom tokenized sequences and position ids."""

    data_class = DatasetClass(None,
                            fim_strategy=fim_strategy,
                            mask_fraction=mask_fraction,
                            max_patches=max_patches,
                            add_position_ids=add_position_ids)
    if len(context_tokens) >= 1:
        seq, pos_ids = data_class.sample_sequences(context_tokens.numpy()[0], num_sequences=num_sequences, shuffle=shuffle)
        # convert to tensor and add batch dimension
        seq = torch.asarray(seq, dtype=torch.int64)[None,:]
        pos_ids = torch.asarray(pos_ids, dtype=torch.int64)[None,:]
        seq = torch.cat([seq, target_tokens], dim=1)
        pos_ids = torch.cat([pos_ids, target_pos_ids], dim=1)
        return seq, pos_ids
    else:
        return target_tokens, target_pos_ids

def prepare_target(target, use_fim=None):
    """Prepare the target sequence for the model using a custom tokenized sequence.
    use_fim is a dictionary with the positions that should be masked.
    use_fim = {"<cls>": 1} _-> start to generate autoregressively from 1st position
    use_fim = {"<cls>": 10} -> start to generate autoregressively from 10th position
    use_fim = {"<mask-1>": ((10,13), 6)} -> mask positions from 10 to 13 (i.e. 10,11,12) and fill it with 6 tokens,
    use_fim = {"<mask-1>": ((10,13), 6), "<mask-2>": ((15,20), 2)} -> mask positions from 10 to 13 and 15 to 20 and fill it with 6 and 2 tokens
    """
    if "<cls>" in use_fim:
        target = target[:,:use_fim["<cls>"]]
        pos_ids = torch.arange(target.shape[1], dtype=torch.int64)[None,:]
        assert target.shape[1] == pos_ids.shape[1]
        return target, pos_ids
    else:
        is_fim_dict = {}
        pos_ids = torch.arange(target.shape[1], dtype=torch.int64)[None,:] # default position ids
        diff_length = 0
        for mask in use_fim:
            assert "mask" in mask
            mask_positions, length = use_fim[mask]
            # update mask_positions to take into account the inserted parts
            mask_positions = (mask_positions[0]-diff_length, mask_positions[1]-diff_length)
            diff_length = mask_positions[1] - mask_positions[0]
            new_target = torch.cat([target[:,:mask_positions[0]],
                                    torch.full((target.shape[0], 1), AA_TO_ID[mask], dtype=torch.int64),
                                    target[:,mask_positions[1]:]], dim=1)
            new_pos_ids = torch.cat([pos_ids[:,:mask_positions[0]+1],
                                    pos_ids[:,mask_positions[1]:]+length-diff_length], dim=1)
            is_fim_dict[AA_TO_ID[mask]] = pos_ids[:,mask_positions[0]].squeeze().item()
            target = new_target
            pos_ids = new_pos_ids
            diff_length -= 1

        new_target = torch.cat([target,
                                torch.full((target.shape[0], 1), AA_TO_ID["<eos>"], dtype=torch.int64)], dim=1)
        new_pos_ids = torch.cat([pos_ids,
                                torch.full((target.shape[0], 1), 0, dtype=torch.int64)], dim=1)
        assert new_target.shape[1] == new_pos_ids.shape[1]
        return new_target, new_pos_ids, is_fim_dict

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing.event_accumulator import ScalarEvent
from torch.utils.tensorboard import SummaryWriter
import glob
import matplotlib.pyplot as plt

def load_tensorboard_data(path):
    """
    Load the TensorBoard data
    """
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    # Assuming you're interested in scalar summaries; adjust if otherwise
    # list all tags
    tags = ea.Tags()['scalars']
    
    scalars = {tag: ea.scalars.Items(tag) for tag in tags}
    return scalars

def filter_datapoints(scalars, condition):
    """
    Filter out the datapoint you want to delete (customize this logic)
    """
    # Example condition: lambda x: x.step != step_to_delete
    return {tag: [s for s in scalars if not condition(s)] for tag, scalars in scalars.items()}

def save_to_tensorboard(filtered_data, output_path):
    """
    Save modified data back to a new TensorBoard file
    """
    writer = SummaryWriter(output_path) 
    for tag, scalars in filtered_data.items():
        for data in scalars:
            writer.add_scalar(tag, data.value, global_step=data.step, walltime=data.wall_time)

def merge_loggings(directory, output_path, plot_metric=None):
    """
    Merge all the TensorBoard files in a directory into a single file.
    Keeps only the metrics with latest wall time for each step (i.e. the last logged value for each step)
    """
    # Find all the TensorBoard files
    def find_files(directory, pattern):
        return glob.glob(f"{directory}/{pattern}")
    all_paths = find_files(directory, "events.out.tfevents.*")
    # Merge all the data
    best_wall_times = {}
    updated_metrics = {}
    for elem in all_paths:
        try:
            # Load the data from one logging
            scalars = load_tensorboard_data(elem)
            # Make a dictionary with step number as key
            all_metrics = {k: {s.step: s for s in scalars[k]} for k in scalars.keys()}
            if plot_metric is not None:
                plt.plot([s.wall_time for s in scalars[plot_metric]], [s.value for s in scalars[plot_metric]])
            # iterate over all the metrics
            for k in all_metrics.keys():
                if k not in updated_metrics.keys():
                    updated_metrics[k] = {}
                if k not in best_wall_times:
                    best_wall_times[k] = {}
                # Get wall time of each step
                steps_time = {step: s.wall_time for step,s in all_metrics[k].items()}
                # iterate over steps and pick only the metrics associated with the best wall time
                for key, value in steps_time.items():
                    if key not in updated_metrics[k]:
                        best_wall_times[k][key] = value
                        updated_metrics[k][key] = all_metrics[k][key]
                    elif value > best_wall_times[k][key]:
                        best_wall_times[k][key] = value
                        updated_metrics[k][key] = all_metrics[k][key]
                    else:
                        continue
        except:
            print("Could not load.\t", elem.split("/")[-1])
    # Sort the metrics by step
    new_logging = {k: list(updated_metrics[k].values()) for k in updated_metrics.keys()}
    new_logging = {k: sorted(v, key=lambda x: x.step) for k, v in new_logging.items()}
    # Save the merged data
    save_to_tensorboard(new_logging, output_path)
    plt.title(f"Metric: {plot_metric}")
    plt.show()
    return new_logging
    
def concatenate_loggings(logging1_path, logging2_path, step_range1, step_range2, output_path):
    """
    Concatenate the two loggings, assuming they have the same metrics. Use steps from step_range1[0] to step_range1[1]
    for logging1 and from step_range2[0] to step_range2[1] for logging2.
    Change the step numbers of logging2 to be continuous with logging1. and verify that the steps taken in each logging
    are the ones specified by step_range1 and step_range2.
    """
    logging1 = load_tensorboard_data(logging1_path)
    logging2 = load_tensorboard_data(logging2_path)
    if step_range1 is None:
        k = list(logging1.keys())[0]
        step_range1 = (logging1[k][0].step, logging1[k][-1].step)
    if step_range2 is None:
        k = list(logging2.keys())[0]
        step_range2 = (logging2[k][0].step, logging2[k][-1].step)
    new_logging = {}
    for key in logging1.keys():
        new_logging[key] = [el for el in logging1[key] if el.step >= step_range1[0] and el.step < step_range1[1]]
        for el in logging2[key]:
            if el.step >= step_range2[0] and el.step <= step_range2[1]:
                # not possible to assign to the step attribute of the object, make a new ScalarEvent object identical
                # to el but with the step attribute changed
                new_step_value = el.step - step_range2[0] + step_range1[1]
                new_el = ScalarEvent(step=new_step_value, wall_time=el.wall_time, value=el.value)        
                
                new_logging[key].append(new_el)
        new_logging[key] = sorted(new_logging[key], key=lambda x: x.step)
    save_to_tensorboard(new_logging, output_path)
    return new_logging

# %% ../nbs/99_utils.ipynb 6
def print_number_of_parameters(model):
    print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

def find_fim_indices(is_cls_tokens, is_eos_tokens):
    """Function to find the indices of the FIM tokens in the sequences.
    """
    # add a cls token at the beginning
    is_cls_tokens = torch.cat([torch.ones_like(is_cls_tokens[:, :1]), is_cls_tokens], dim=1)
    is_eos_tokens = torch.cat([torch.zeros_like(is_eos_tokens[:, :1]), is_eos_tokens], dim=1)
    # both eos and cls tokens
    bol = is_cls_tokens | is_eos_tokens
    tmp = torch.zeros_like(is_cls_tokens, dtype=torch.int)
    tmp[torch.nonzero(is_cls_tokens, as_tuple=True)] = 1
    tmp[torch.nonzero(is_eos_tokens, as_tuple=True)] = -1
    bol1 = torch.clone(bol)
    for batch_ind in range(tmp.size(0)):
        tmp1 = tmp[batch_ind,bol[batch_ind]]
        # find all positions where a 1 if preceeded by a -1
        tmp1 = tmp1[:-1]*tmp1[1:]
        # add the first element to make the sequence start with a 1
        tmp1 = torch.cat([torch.ones_like(tmp1[:1]).to(tmp1.device), tmp1])
        new_bol = tmp1<0
        # bool array True only in the positions where a 1 is preceeded by a -1
        bol1[batch_ind,bol[batch_ind]] = False if new_bol.size(0) == 0 else new_bol
    cumulative_sum = torch.cumsum(bol1, dim=1)
    # Use modulo operation to get the desired tensor
    bol2 = cumulative_sum % 2 == 1
    bol2[is_eos_tokens]= False
    return bol2[:,1:]

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions).permute(0, 2, 1)
    labels = torch.tensor(labels)
    # shift labels to align them with predictions and remove last prediction to match the length
    predictions = predictions[:, :, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    # compute unreduced elementwise loss
    unreduced_loss = torch.nn.functional.cross_entropy(predictions, labels, reduction="none")
    # compute reconstruction accuracy
    reconstruction = (predictions.argmax(1) == labels)

    # start and end tokens
    is_cls_tokens = (labels == AA_TO_ID["<cls>"])
    is_eos_tokens = (labels == AA_TO_ID["<eos>"])
    # fill in the middle tokens
    if False:
        fim_tokens = torch.zeros(is_cls_tokens.size(0), is_cls_tokens.size(1), dtype=torch.bool)
        in_mask_vector = torch.zeros(is_cls_tokens.size(0), dtype=torch.bool)
        for j in range(is_cls_tokens.size(1)):
            in_mask_vector = in_mask_vector & ~is_cls_tokens[:, j]
            fim_tokens[:, j] = in_mask_vector
            in_mask_vector = in_mask_vector | is_eos_tokens[:, j]
    fim_tokens = find_fim_indices(is_cls_tokens, is_eos_tokens)
        
    number_sequences = torch.cumsum(torch.cat([torch.zeros(is_cls_tokens.size(0),1, dtype=torch.int32), is_cls_tokens[:,:-1]],1), -1)
    # fist, second and last sequence tokens
    first_sequence_tokens = ((~fim_tokens & (labels < 33)) | fim_tokens) & (number_sequences == 0)
    second_sequence_tokens = ((~fim_tokens & (labels < 33)) | fim_tokens) & (number_sequences == 1)
    last_sequence_tokens = ((~fim_tokens & (labels < 33)) | fim_tokens) & (number_sequences == (number_sequences.max(1).values[:, None] - 1))
    # end of mask tokens
    end_of_masks = (fim_tokens & (labels > 33)) | is_cls_tokens | is_eos_tokens
    return {"loss/all": torch.mean(unreduced_loss).item(),
            "loss/end_span": torch.mean(unreduced_loss[end_of_masks]).item(),
            "perplexity/seq": torch.mean(torch.exp(torch.mean(unreduced_loss, dim=1))).item(),
            "perplexity/end_span": torch.exp(torch.mean(unreduced_loss[end_of_masks])).item(),
            "perplexity/batch": torch.exp(torch.mean(unreduced_loss)).item(),
            "perplexity/first_seq": torch.exp(torch.mean(unreduced_loss[first_sequence_tokens])).item(),
            "perplexity/second_seq": torch.exp(torch.mean(unreduced_loss[second_sequence_tokens])).item(),
            "perplexity/last_seq": torch.exp(torch.mean(unreduced_loss[last_sequence_tokens])).item(),
            "perplexity/fim": torch.exp(torch.mean(unreduced_loss[fim_tokens])).item(),
            "reconstruction/all": torch.mean(reconstruction.float()).item(),
            "reconstruction/end_span": torch.mean(reconstruction[end_of_masks].float()).item(),
            "reconstruction/first_seq": torch.mean(reconstruction[first_sequence_tokens].float()).item(),
            "reconstruction/second_seq": torch.mean(reconstruction[second_sequence_tokens].float()).item(),
            "reconstruction/last_seq": torch.mean(reconstruction[last_sequence_tokens].float()).item(),
            "reconstruction/fim": torch.mean(reconstruction[fim_tokens].float()).item(),}

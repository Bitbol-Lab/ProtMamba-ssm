from .utils import AA_TO_ID
from .fim import NoFIM, SingleSpanFIM, MultipleSpanFIM
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, Sequence

# Make dataset
class Uniclust30_Dataset(Dataset):
    """
        Dataset class used to import the Uniclust30 folders.
        If `filename` = "encoded_MSAs.pkl", it will load the full dataset.
        If `filename` = "encoded_MSAs_subset.pkl", it will load a small subset of the dataset.
        If `sample` = True, it will sample a random number of sequences from each cluster.
        If `sample` = False, it will load all the sequences from each cluster (and shuffle them).
        To limit the length of the MSAs, set `max_msa_len` to a positive integer.
        If `reverse` = True, it will reverse the sequences with probability 0.5 and move the last token to the front.
        If `scrambling_strategy` = "no-scramble", it will not scramble the sequences and simply concatenate them.
        If `scrambling_strategy` = "OpenAI", it will scramble the sequences using the OpenAI strategy.
        If `scrambling_strategy` = "inpaint", it will scramble the sequences using the inpaint strategy. In this case it will use
        `max_patches` patches and mask `mask_fraction` of the patches.
    """
    _FIM = {"no-scramble": NoFIM, "one_span": SingleSpanFIM, "multiple_span": MultipleSpanFIM}
    _POSIDS = {"none", "1d", "2d"}

    def __init__(self, filename="encoded_MSAs_train.pkl",
                 filepath="/nvme1/common/OpenProteinSet/",
                 sample=False,
                 max_msa_len=-1,
                 reverse=False,
                 seed=42,
                 troubleshoot=False,
                 fim_strategy="no-scramble",
                 max_patches=5,
                 mask_fraction=0.2,
                 always_mask=False,
                 max_position_embeddings=2048,
                 max_seq_position_embeddings=512,
                 add_position_ids="none", ):
        np.random.seed(seed)
        self.path = filepath
        # self.path_clusters = self.path + "OpenProteinSet_uniclust30-filtered/"
        if filename:
            self.dataset = pickle.load(open(self.path + filename, "rb"))
            self.cluster_names = list(self.dataset.keys())
        else:
            self.dataset = None
            self.cluster_names = []
        self.sample = sample
        self.max_msa_len = max_msa_len
        self.reverse = reverse
        self.fim_strategy = fim_strategy
        if fim_strategy in Uniclust30_Dataset._FIM:
            self.fim = Uniclust30_Dataset._FIM[fim_strategy](max_patches=max_patches,
                                                             mask_fraction=mask_fraction,
                                                             always_mask=always_mask,
                                                             add_position_ids=add_position_ids != "none",
                                                             troubleshoot=troubleshoot)
        else:
            raise ValueError(f'Fill in the middle stragy "{fim_strategy}" not recognized.')
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_position_embeddings = max_seq_position_embeddings
        self.add_position_ids = add_position_ids

        self.troubleshoot = troubleshoot

    def __len__(self):
        return len(self.cluster_names)

    def __getitem__(self, idx, shuffle=True):
        # get all the sequences in the cluster
        sequences = self.get_sequences(idx)
        # get total number of sequences in the cluster and choose how many to sample
        orig_num_sequences = len(self.get_index_start_of_sequences(sequences))
        num_sequences = np.random.randint(1, orig_num_sequences + 1) if self.sample else orig_num_sequences
        # sample the sequences
        sequences, position_ids = self.sample_sequences(sequences, num_sequences, shuffle=shuffle)
        # with probability 0.5, reverse the sequences and move the last token to the front
        sequences, position_ids = self.reverse_sequences(sequences, position_ids) if (
                self.reverse and np.random.rand() > 0.5) else sequences, position_ids
        # limit the length of the MSA
        sequences = sequences[:self.max_msa_len] if self.max_msa_len > 0 else sequences
        if self.add_position_ids != "none":
            position_ids = position_ids[:self.max_msa_len] if self.max_msa_len > 0 else position_ids
        # convert to tensor
        sequences = torch.asarray(sequences, dtype=torch.int64)
        position_ids = torch.asarray(position_ids, dtype=torch.int64).clamp(0,
                                                                            self.max_position_embeddings - 1) if self.add_position_ids!="none" else None

        if self.troubleshoot:
            print(
                f"Cluster {idx} has {orig_num_sequences} sequences, of which {num_sequences} sampled now. Total MSA length: {len(sequences)}")
        if self.add_position_ids == "1d":
            return dict(input_ids=sequences, position_ids=position_ids, labels=sequences)
        if self.add_position_ids == "2d":
            seq_position_ids = (sequences == AA_TO_ID["<cls>"]).int().cumsum(-1).clamp(0,
                                                                                       self.max_seq_position_embeddings - 1).contiguous()
            return dict(input_ids=sequences, position_ids=position_ids, seq_position_ids=seq_position_ids,
                        labels=sequences)
        return dict(input_ids=sequences, labels=sequences)

    def get_sequences(self, idx):
        """Get the sequences in the cluster with index `idx`."""
        cluster_name = self.cluster_names[idx]
        sequences = self.dataset[cluster_name]
        return sequences

    def get_index_start_of_sequences(self, sequences):
        """Get the positions of the start of each sequence in the cluster."""
        return np.where(sequences == 0)[0]

    def reverse_sequences(self, sequence, position_ids=None):
        """Reverse the sequences and move the last token to the front."""
        sequence = sequence[::-1]
        if position_ids is not None:
            position_ids = position_ids[::-1]
        return np.concatenate([sequence[-1:], sequence[:-1]]), np.concatenate(
            [position_ids[-1:], position_ids[:-1]]) if position_ids is not None else None

    def sample_sequences(self, sequences, num_sequences, shuffle=True, which_seqs=None):
        """Sample `num_sequences` from the sequences in the cluster."""
        L = len(sequences)
        # get the indexes of the start of each sequence
        inds = self.get_index_start_of_sequences(sequences)
        # check that there are sequences in the cluster and that there are enough of them
        assert len(inds) > 0, "No sequences found in cluster."
        assert len(inds) >= num_sequences, "Not enough sequences in cluster."
        # sample n_sequences randomly from the sequences
        if which_seqs is None:
            if shuffle:
                which_seqs = np.random.choice(np.arange(len(inds)), num_sequences, replace=False)
            else:
                which_seqs = np.arange(len(inds))[-num_sequences:]
        # get the tuples of start and end indexes of the sequences
        tuples = [(inds[i], inds[i + 1]) if i < len(inds) - 1 else (inds[i], L) for i in which_seqs]
        if self.troubleshoot:
            print(f"Sampled sequences: {tuples}")
        # concatenate the sequences
        sequences, position_ids = self.fim.apply(sequences, tuples)
        return sequences, position_ids


def make_dataloader(dataset):
    """Basic function to make a dataloader.
    """
    dataloader = DataLoader(dataset)

@dataclass
class DataCollatorForUniclust30Dataset(object):
    """
    Collate examples into a batch, and pad batch to the maximum sequence length.
    """

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "input_ids")) 
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=AA_TO_ID["<pad>"])
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        if "seq_position_ids" in instances[0] and "position_ids" in instances[0]:
            position_ids = torch.nn.utils.rnn.pad_sequence(
                [instance["position_ids"] for instance in instances],
                batch_first=True, padding_value=0)
            seq_position_ids = torch.nn.utils.rnn.pad_sequence(
                [instance["seq_position_ids"] for instance in instances],
                batch_first=True, padding_value=0)
            return dict(
                input_ids=input_ids,
                labels=labels,
                position_ids=position_ids,
                seq_position_ids=seq_position_ids,
                attention_mask=input_ids.ne(AA_TO_ID["<pad>"]),
            )

        if "position_ids" in instances[0]:
            position_ids = torch.nn.utils.rnn.pad_sequence(
                [instance["position_ids"] for instance in instances],
                batch_first=True, padding_value=0)
            return dict(
                input_ids=input_ids,
                labels=labels,
                position_ids=position_ids,
                attention_mask=input_ids.ne(AA_TO_ID["<pad>"]),
            )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(AA_TO_ID["<pad>"]),
        )

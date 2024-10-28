import hashlib
import math
import os
import pickle
from pathlib import Path
from typing import Optional, Callable, Union

import numpy as np
import torch
from datasets import Dataset

from ProtMamba_ssm import AA_TO_ID, NoFIM, SingleSpanFIM, MultipleSpanFIM


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

    def __getitem__(self, idx):
        # get all the sequences in the cluster
        sequences = self.get_sequences(idx)
        # get total number of sequences in the cluster and choose how many to sample
        orig_num_sequences = len(self.get_index_start_of_sequences(sequences))
        num_sequences = np.random.randint(1, orig_num_sequences + 1) if self.sample else orig_num_sequences
        # sample the sequences
        sequences, position_ids = self.sample_sequences(sequences, num_sequences)
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

    def sample_sequences(self, sequences, num_sequences, shuffle=True, which_seqs: Union[np.array, None]=None):
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


def compute_hamming_csim_torch(
        seqs: torch.Tensor,
        ungapped_msa: torch.Tensor,
        gap_token: int,
        gap_token_mask: int,
) -> torch.Tensor:
    return (seqs.unsqueeze(1) == ungapped_msa).sum(dim=2)

def _compute_homology_weights(
        ungapped_msa: np.ndarray,
        gap_token: int,
        gap_token_mask: int,
        theta: float,
        hamming_csim_func: Callable,
        max_memory: int = 20,
        can_use_torch: bool = True,
) -> np.ndarray:
    use_torch = can_use_torch and torch.cuda.is_available()
    if use_torch:
        hamming_csim_func = compute_hamming_csim_torch
    batch_size = math.floor(
        2
        * 1024
        * 1024
        * 1024
        / (ungapped_msa.shape[0] * ungapped_msa.shape[1])
        * max_memory
        / 40
    )

    batch_size = 1 if batch_size == 0 else batch_size

    neighbors = []
    if not use_torch:
        masked_ungapped_msa = ungapped_msa.copy()
    else:
        ungapped_msa = torch.from_numpy(ungapped_msa).byte().cuda()
        masked_ungapped_msa = ungapped_msa.clone()
    masked_ungapped_msa[masked_ungapped_msa == gap_token] = gap_token_mask
    for b_start in range(0, len(ungapped_msa), batch_size):
        b_end = b_start + batch_size
        seqs = ungapped_msa[b_start:b_end]

        sim = hamming_csim_func(
            seqs=seqs,
            ungapped_msa=masked_ungapped_msa,
            gap_token=gap_token,
            gap_token_mask=gap_token_mask,
        )
        if not use_torch:
            sim = sim / (seqs != gap_token).sum(axis=1, keepdims=True)
            d = 1 - sim
            d = d.clamp(0, 1)
            this_neighbors = (d <= theta).sum(axis=1)
        else:
            sim = sim / (seqs != gap_token).sum(dim=1, keepdim=True)
            d = 1 - sim
            # fillna
            d[torch.isnan(d)] = 0
            d = d.clamp(0, 1)
            this_neighbors = (d <= theta).sum(dim=1).cpu()
        neighbors.append(this_neighbors)
    return np.concatenate(neighbors)


def compute_homology_weights(
        ungapped_msa: np.ndarray,
        theta: float = 0.2,
        gap_token: int = AA_TO_ID["-"],
        gap_token_mask: int = 255,
        hamming_csim_func: Callable = compute_hamming_csim_torch,
) -> tuple[int, np.ndarray]:
    """
    Calculate the effective number of sequences and sampling probability for the NEIGHBORS and NEIGHBORS_NO_LIMIT sampling methods using numpy.

    Parameters:

        ungapped_msa (np.ndarray): The MSA (from .fa).
        theta (float, optional): A parameter used to determine the similarity between sequences. Default is 0.2.
        gap_token (int, optional): The token representing gaps in the (Uniprot21 encoded) MSA. Default is 20.
        gap_token_mask (int): token for masking gaps. should be a token not representing any other value.

    Returns:

        tuple[int, np.ndarray]: A tuple containing the effective number of sequences and the sampling probability for each sequence in the MSA.
    """
    neighbors = _compute_homology_weights(
        ungapped_msa=ungapped_msa,
        gap_token=gap_token,
        gap_token_mask=gap_token_mask,
        theta=theta,
        hamming_csim_func=hamming_csim_func,
    )
    n_eff = np.sum(1 / neighbors)

    p = 1 / neighbors
    p /= np.sum(p)
    return n_eff, p


class MSASampler:

    def __init__(self, max_similarity, max_dissimilarity, force_include_first=True):
        self.max_similarity = max_similarity
        self.max_dissimilarity = max_dissimilarity
        self.force_include_first = force_include_first
        self.theta = 0.2

    def _get_sim_filtered_idxs(self, msa: np.ndarray) -> np.ndarray:
        nonnormalized_sim = (msa == msa[[0]]).sum(axis=1)
        normfactor = msa.shape[1]
        norm_sim = nonnormalized_sim / normfactor

        assert (norm_sim.min() >= 0) and (norm_sim.max() <= 1)
        dsim = 1 - norm_sim

        max_sim_filter = norm_sim <= self.max_similarity
        max_dissim_filter = dsim <= self.max_dissimilarity
        return np.where(max_sim_filter & max_dissim_filter)[0]

    def get_weights(
            self, msa: np.ndarray,
    ) -> tuple[Optional[float], Optional[np.ndarray]]:
        return compute_homology_weights(
            ungapped_msa=msa,
            theta=self.theta,
            gap_token_mask=255,

        )

    def get_sample_idxs(
            self,
            msa: np.ndarray,
            name: str = None,
            size: int = 1,
            random = False,
            result_cache_dir: Optional[str] = "/data2/malbrank/protein_gym/cache",
    ) -> np.ndarray:
        if random:
            return np.random.choice(len(msa), replace=False, size=size) if len(msa) >= size else np.arange(len(msa))
        msa = np.array([[AA_TO_ID[aa] for aa in seq.upper()][:len(msa[0])] for seq in msa], dtype=np.uint8)

        if name is not None and name+".npy" in os.listdir(result_cache_dir):
            weights = np.load(os.path.join(result_cache_dir, name+".npy"))
        elif name is not None:
            _, weights = self.get_weights(
                msa=msa,
            )
            np.save(os.path.join(result_cache_dir, name), weights)
        else:
            _, weights = self.get_weights(
                msa=msa,
            )


        original_msa_sample_idxs = np.arange(len(msa))
        sample_idxs = self._get_sim_filtered_idxs(msa)
        original_msa_sample_idxs = original_msa_sample_idxs[sample_idxs]

        if self.force_include_first:
            original_msa_sample_idxs = np.concatenate(
                [[0], original_msa_sample_idxs[original_msa_sample_idxs != 0]]
            )
        return np.random.choice(len(msa), replace=False, size=size, p=weights / weights.sum()) if len(msa) >= size else original_msa_sample_idxs
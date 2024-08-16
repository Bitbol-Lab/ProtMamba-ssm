import logging
import math
import os
import random
import shutil
import subprocess
import tarfile
import time
from typing import List

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_mmseqs2_api(x, prefix, use_env=True, use_filter=True, filter=None, host_url="https://api.colabfold.com",
                    user_agent: str = "") -> List[str]:
    submission_endpoint = "ticket/msa"

    headers = {}
    if user_agent != "":
        headers['User-Agent'] = user_agent
    else:
        logger.warning(
            "No user agent specified. Please set a user agent (e.g., 'toolname/version contact@email') to help us debug in case of problems. This warning will become an error in the future.")

    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        while True:
            error_count = 0
            try:
                # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                # "good practice to set connect timeouts to slightly larger than a multiple of 3"
                res = requests.post(f'{host_url}/{submission_endpoint}', data={'q': query, 'mode': mode}, timeout=6.02,
                                    headers=headers)
            except requests.exceptions.Timeout:
                logger.warning("Timeout while submitting to MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break

        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID):
        while True:
            error_count = 0
            try:
                res = requests.get(f'{host_url}/ticket/{ID}', timeout=6.02, headers=headers)
            except requests.exceptions.Timeout:
                logger.warning("Timeout while fetching status from MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID, path):
        error_count = 0
        while True:
            try:
                res = requests.get(f'{host_url}/result/download/{ID}', timeout=6.02, headers=headers)
            except requests.exceptions.Timeout:
                logger.warning("Timeout while fetching result from MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        with open(path, "wb") as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # compatibility to old option
    if filter is not None:
        use_filter = filter

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    # define path
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f'{path}/out.tar.gz'
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = []
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]
    # lets do it!
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    raise Exception(
                        f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')
                if out["status"] == "MAINTENANCE":
                    raise Exception(f'MMseqs2 API is undergoing maintenance. Please try again in a few minutes.')

                # wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    raise Exception(
                        f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')
            download(ID, tar_gz_file)

    a3m_files = [f"{path}/uniref.a3m"]

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file, "r"):
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                a3m_lines[M].append(line)
    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]
    return a3m_lines


def full_results_to_individuals(fasta_file, result_file, outdir: str):
    r""" Split the full results into individual a3m files

    Args:
        fasta_file (str): Path to reference file
        result_file (str): Path to result file
        outdir (str): Output directory
    """
    labels = []
    sequences = []
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                labels.append("_".join(line[1:].strip().split("_")[:2]))
                sequences.append("")
            else:
                sequences[-1] += line.strip()
    seqs_unique = []
    [seqs_unique.append(x) for x in sequences if x not in seqs_unique]
    Ms = [seqs_unique.index(sequences) for sequences in seqs]
    seqid_to_labels = {i: [] for i in range(len(seqs_unique))}
    for i, label in zip(Ms, labels):
        seqid_to_labels[i].append(label)

    is_first_seq = True
    filenames = []
    with open(result_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                label = line[1:].strip()
                if len(label) == 3:
                    idx = int(label) - 101
                    filenames = seqid_to_labels[idx]
                    first_seq = seqs_unique[idx]
                    for filename in filenames:
                        with open(f"{outdir}/{filename}.a3m", "w") as f:
                            f.write(f">{filename}\n")
                            f.write(first_seq + "\n")
                    is_first_seq = True
                else:
                    for filename in filenames:
                        with open(f"{outdir}/{filename}.a3m", "a") as f:
                            f.write(line)
            else:
                if is_first_seq:
                    is_first_seq = False
                else:
                    for filename in filenames:
                        with open(f"{outdir}/{filename}.a3m", "a") as f:
                            f.write(line)


if __name__ == "__main__":

    user_agent = ... # str: User agent to use for the API requests
    fasta_file = ... # str: Path to the fasta file that contain the sequences to query
    result_file = ... # str: Path to the result file that will contain the a3m results
    temp_folder = ... # str: Path to the temporary folder to store the intermediate files
    seqs = []
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                seqs.append("")
            else:
                seqs[-1] += line.strip()
    a3m_lines = run_mmseqs2_api(seqs, temp_folder, use_env=True, use_filter=True,
                                filter=None, host_url="https://api.colabfold.com", user_agent=user_agent)
    with open(result_file, "w") as f:
        for line in a3m_lines:
            f.write(line)
            f.write("\n")
    output_dir = "/data2/malbrank/protein_gym/mmseqs_colabfold_protocol"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    full_results_to_individuals(fasta_file, result_file, output_dir)

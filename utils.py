import logging
import os
import re
import sys

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(out_dir):
    logger = logging.getLogger("Exp")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


class CharLabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.char2idx = {char: idx for idx, char in enumerate(alphabet)}
        self.idx2char = {idx: char for idx, char in enumerate(alphabet)}

    def encode(self, labels):
        # labels: list of single-character strings
        return torch.tensor(
            [self.char2idx[label] for label in labels], dtype=torch.long
        )

    def decode(self, indices):
        # indices: tensor of shape (B,)
        return [self.idx2char[idx.item()] for idx in indices]


def format_string_for_wer(str):
    str = re.sub("([\[\]{}/\\()\"'&+*=<>?.;:,!\-—_€#%°])", r" \1 ", str)
    str = re.sub("([ \n])+", " ", str).strip()
    return str

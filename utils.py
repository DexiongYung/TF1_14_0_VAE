# -*- coding: future_fstrings -*-
import matplotlib
matplotlib.use('Agg')
import collections
import string
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os import path


def load_data(n, SOS='[', EOS=']'):
    df = pd.read_csv(n)
    names = df['name'].tolist()
    name_probs = df['probs'].tolist()
    chars = string.ascii_letters + SOS + EOS
    c_to_n_vocab = dict(zip(chars, range(len(chars))))
    n_to_c_vocab = dict(zip(range(len(chars)), chars))

    sos_idx = c_to_n_vocab[SOS]
    eos_idx = c_to_n_vocab[EOS]

    return names, name_probs, c_to_n_vocab, n_to_c_vocab, sos_idx, eos_idx


def create_batch(all_names, probs_list, batch_size, vocab, SOS, EOS):
    num_names = len(all_names)
    samples = np.random.choice(num_names,
                               batch_size,
                               p=probs_list / np.sum(probs_list))
    names = [all_names[samples[i]] for i in range(batch_size)]

    seq_length = len(max(names, key=len)) + 1

    # Names length should be length of the name + SOS xor EOS
    names_length = [len(n) + 1 for n in names]
    names_input = [(SOS + s).ljust(seq_length, EOS) for s in names]
    names_input = [list(map(vocab.get, s)) for s in names_input]
    names_output = [(s).ljust(seq_length, EOS) for s in names]
    names_output = [list(map(vocab.get, s)) for s in names_output]

    return names_input, names_output, names_length


def plot_losses(losses, folder="plot", filename="checkpoint.png"):
    if not path.exists(folder):
        os.mkdir(folder)

    x = list(range(len(losses)))
    plt.plot(x, losses, 'b--', label="Unsupervised Loss")
    plt.title("Loss Progression")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()

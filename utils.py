import collections
import string
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os import path


def load_data(n):
    df = pd.read_csv(n)
    names = df['name'].tolist()
    seq_length = df['name'].str.len().max()

    SOS = '['
    EOS = ']'
    chars = string.ascii_letters + SOS + EOS

    vocab = dict(zip(chars, range(len(chars))))

    length = np.array([len(n)+1 for n in names])
    names_input = [(SOS+s).ljust(seq_length, EOS) for s in names]
    names_output = [s.ljust(seq_length, EOS) for s in names]
    names_input = np.array([np.array(list(map(vocab.get, s)))
                            for s in names_input])
    names_output = np.array([np.array(list(map(vocab.get, s)))
                             for s in names_output])
    prop = np.array([] for i in range(len(names)))
    return names_input, names_output, chars, vocab, prop, length


def plot_losses(losses, folder: str = "plot", filename: str = "checkpoint.png"):
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

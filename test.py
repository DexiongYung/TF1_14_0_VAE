from model import VAE
from utils import *
import numpy as np
import os
import json
import tensorflow as tf
import time

class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])
        
NAME = 'first'
json_file = json.load(open(f'json/{NAME}.json', 'r'))
args = Dict2Obj(json_file)

model = VAE(len(args.vocab), args)
model.restore(f'{args.save_dir}/{args.name}.ckpt')

# convert names to numpy array
names, name_probs, c_to_n_vocab, n_to_c_vocab, sos_idx, eos_idx = load_data(
    args.prop_file)

SOS = n_to_c_vocab[sos_idx]
EOS = n_to_c_vocab[eos_idx]

names_input, names_output, names_length = create_batch(names, name_probs, 128, c_to_n_vocab, SOS, EOS)

x = np.array(names_input)
y = np.array(names_output)
l = np.array(names_length)
pred, cost = model.test(x, y, l)

names_test = [list(map(n_to_c_vocab.get, s)) for s in pred.tolist()]
names_input = [list(map(n_to_c_vocab.get, s)) for s in names_input]
for i in range(len(names_test)):
    name = names_test[i]
    name = ''.join(name)
    original = ''.join(names_input[i])
    print(f'Original: {original}, Recon: {name}')

# -*- coding: future_fstrings -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from model import VAE
from utils import *
import tensorflow as tf
import numpy as np
import argparse
import time
import json


parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Session name', type=str, default='first')
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument('--unit_size',
                    help='unit_size of rnn cell',
                    type=int,
                    default=512)
parser.add_argument('--n_rnn_layer',
                    help='number of rnn layer',
                    type=int,
                    default=3)
parser.add_argument('--prop_file', help='name of property file', type=str, default='data/first.csv')
parser.add_argument('--mean', help='mean of VAE', type=float, default=0.0)
parser.add_argument('--stddev', help='stddev of VAE', type=float, default=1.0)
parser.add_argument('--num_epochs', help='epochs', type=int, default=1000)
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
parser.add_argument('--save_every',
                    help='Number of iterations before saying',
                    type=int,
                    default=300)
parser.add_argument('--save_dir', help='save dir', type=str, default='save/')
parser.add_argument('--continue_train',
                    help='Continue session name training?', type=bool, default=False)
args = parser.parse_args()

# convert names to numpy array
names, name_probs, c_to_n_vocab, n_to_c_vocab, sos_idx, eos_idx = load_data(args.prop_file)

SOS = n_to_c_vocab[sos_idx]
EOS = n_to_c_vocab[eos_idx]

model = VAE(len(c_to_n_vocab), args)
print('Number of parameters : ',
      np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

if args.continue_train:
    json_file = json.load(open(f'json/{args.name}.json', 'r'))
    t_args = argparse.Namespace()
    t_args.__dict__.update(json_file)
    args = parser.parse_args(namespace=t_args)

    model.restore(f'{args.save_dir}/{args.name}.ckpt')
else:
    args.vocab = c_to_n_vocab
    args.sos_idx = sos_idx
    args.eos_idx = eos_idx

    # make save_dir
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    if not path.exists('json'):
        os.mkdir('json')

    with open(f'json/{args.name}.json', 'w') as f:
        json.dump(vars(args), f)

# divide data into training and test set
num_train_data = int(len(names) * 0.75)
num_test_data = len(names) - num_train_data


for epoch in range(args.num_epochs):
    print('Epoch {}/{}'.format(epoch, args.num_epochs))
    train_loss = []
    test_loss = []
    for iteration in range(num_train_data // args.batch_size):
        x, y, l = create_batch(names, name_probs, args.batch_size,
                               c_to_n_vocab, SOS, EOS)
        x = np.array(x)
        y = np.array(y)
        l = np.array(l)

        try:
            cost = model.train(x, y, l)
        except Exception as e:
            print(e)

        train_loss.append(cost)

        if iteration % args.save_every == 0:
            ckpt_path = args.save_dir + f'/{args.name}.ckpt'
            model.save(ckpt_path)
            plot_losses(train_loss, filename=f'{args.name}_train.png')

    for iteration in range(num_test_data // args.batch_size):
        x, y, l = create_batch(names, name_probs, args.batch_size,
                               c_to_n_vocab, SOS, EOS)
        x = np.array(x)
        y = np.array(y)
        l = np.array(l)

        try:
            _, cost = model.test(x, y, l)
        except Exception as e:
            print(e)

        test_loss.append(cost)

    if epoch%10 == 0:
      plot_losses(test_loss, filename=f'{args.name}_test.png')

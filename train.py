from model import CVAE
from utils import *
import numpy as np
import os
import tensorflow as tf
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Session name', type=str, default='first')
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument(
    '--unit_size', help='unit_size of rnn cell', type=int, default=512)
parser.add_argument(
    '--n_rnn_layer', help='number of rnn layer', type=int, default=3)
parser.add_argument('--prop_file', help='name of property file', type=str)
parser.add_argument('--mean', help='mean of VAE', type=float, default=0.0)
parser.add_argument('--stddev', help='stddev of VAE', type=float, default=1.0)
parser.add_argument('--num_epochs', help='epochs', type=int, default=100)
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
parser.add_argument(
    '--save_every', help='Number of iterations before saying', type=int, default=1000)
parser.add_argument(
    '--num_prop', help='number of propertoes', type=int, default=0)
parser.add_argument('--save_dir', help='save dir', type=str, default='save/')
args = parser.parse_args()

print(args)
# convert names to numpy array
names_input, names_output, char, vocab, labels, length = load_data(
    args.prop_file)
vocab_size = len(char)
inv_vocab = {v: k for k, v in vocab.items()}

# make save_dir
if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

# divide data into training and test set
num_train_data = int(len(names_input)*0.75)
train_names_input = names_input[0:num_train_data]
test_names_input = names_input[num_train_data:-1]

train_names_output = names_output[0:num_train_data]
test_names_output = names_output[num_train_data:-1]

train_length = length[0:num_train_data]
test_length = length[num_train_data:-1]

model = CVAE(vocab_size,
             args
             )
print('Number of parameters : ', np.sum(
    [np.prod(v.shape) for v in tf.trainable_variables()]))

for epoch in range(args.num_epochs):

    st = time.time()
    # Learning rate scheduling
    #model.assign_lr(learning_rate * (decay_rate ** epoch))
    train_loss = []
    test_loss = []
    st = time.time()

    for iteration in range(len(train_names_input)//args.batch_size):
        n = np.random.randint(len(train_names_input), size=args.batch_size)
        x = np.array([train_names_input[i] for i in n])
        y = np.array([train_names_output[i] for i in n])
        l = np.array([train_length[i] for i in n])
        c = np.array([[] for i in n])

        try:
            cost = model.train(x, y, l, c)
        except Exception as e:
            print(e)

        train_loss.append(cost)

        if iteration%args.save_every == 0:
            ckpt_path = args.save_dir+f'/{args.name}.ckpt'
            model.save(ckpt_path, epoch)
            plot_losses(train_loss, filename="train.png")

    for iteration in range(len(test_names_input)//args.batch_size):
        n = np.random.randint(len(test_names_input), size=args.batch_size)
        x = np.array([test_names_input[i] for i in n])
        y = np.array([test_names_output[i] for i in n])
        l = np.array([test_length[i] for i in n])
        c = np.array([[] for i in n])

        try:
            _, cost = model.test(x, y, l, c)
        except Exception as e:
            print(e)

        test_loss.append(cost)

        if iteration % args.save_every == 0:
            plot_losses(train_loss, filename="test.png")

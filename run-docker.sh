#!/bin/bash
docker run --gpus all\
           -v /ubc/cs/research/plai-scratch/BlackBoxML/TF1_14_0_VAE:/TF1_14_0_VAE\
           --rm\
           -it tf114vae:0.2 /bin/bash


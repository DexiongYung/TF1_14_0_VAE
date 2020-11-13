#!/bin/bash

# Run Pipeline
echo "Running in local docker image for training..."
echo "$(which docker)"

docker run --gpus all\
           -v /ubc/cs/research/plai-scratch/BlackBoxML/TF1_14_0_VAE:/TF1_14_0_VAE\
           --rm\
           tf114vae:0.1 /bin/bash\
           -c "cd /TF1_14_0_VAE;\
               python train.py;\
               exit"

echo "Done!"

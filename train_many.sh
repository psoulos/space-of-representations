#!/bin/bash

NUM_PER_ARCH=5

# Relu 10
for ((i=0; i<NUM_PER_ARCH; i++)) {
    python main.py --model_name FullyConnected10ReluDiscriminator --conv False --vae False --hidden_size 10 --activation relu
}

# Relu 100
for ((i=0; i<NUM_PER_ARCH; i++)) {
    python main.py --model_name FullyConnected100ReluDiscriminator --conv False --vae False --hidden_size 100 --activation relu
}

# Relu 200
for ((i=0; i<NUM_PER_ARCH; i++)) {
    python main.py --model_name FullyConnected200ReluDiscriminator --conv False --vae False --hidden_size 200 --activation relu
}

# Sigmoid 10
for ((i=0; i<NUM_PER_ARCH; i++)) {
    python main.py --model_name FullyConnected10SigmoidDiscriminator --conv False --vae False --hidden_size 10 --activation sigmoid
}

# Sigmoid 100
for ((i=0; i<NUM_PER_ARCH; i++)) {
    python main.py --model_name FullyConnected100SigmoidDiscriminator --conv False --vae False --hidden_size 100 --activation sigmoid
}

# Sigmoid 200
for ((i=0; i<NUM_PER_ARCH; i++)) {
    python main.py --model_name FullyConnected200SigmoidDiscriminator --conv False --vae False --hidden_size 200 --activation sigmoid
}

# Convolution 2 features
for ((i=0; i<NUM_PER_ARCH; i++)) {
    python main.py --model_name Conv2Discriminator --conv True --vae False --num_features 2
}

# Convolution 5 features
for ((i=0; i<NUM_PER_ARCH; i++)) {
    python main.py --model_name Conv5Discriminator --conv True --vae False --num_features 5
}

# Convolution 8 features
for ((i=0; i<NUM_PER_ARCH; i++)) {
    python main.py --model_name Conv8Discriminator --conv True --vae False --num_features 8
}

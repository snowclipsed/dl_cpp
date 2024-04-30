#ifndef __MLP_H
#define __MLP_H

#include <vector>
#include "stdio.h"
#include "stdint.h"

#define INPUT_DIM 784
#define OUTPUT_DIM 10
#define HIDDEN_LAYER_SIZE 256
#define NUM_HIDDEN_LAYERS 3


class mlp{

    typedef struct{
        double weights;
        double biases;
        double activations;
        int num_weights;
        
    }network_params;
    
    public:
        mlp();
        ~mlp();
        network_params* *params;
        network_params* init_network();
};

#endif
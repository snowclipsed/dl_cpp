// #include "data_handler.hpp"
// #include "data.hpp"
#include <cstdlib>
#include <vector>
#include "stdio.h"
#include "stdint.h"
#include <ctime>


#define INPUT_DIM 784
#define OUTPUT_DIM 10
#define HIDDEN_LAYER_SIZE 256
#define NUM_HIDDEN_LAYERS 3


class mlp{

    typedef struct{
        double* weights;
        double* biases;
        double* activations;
        int num_weights;
        
    }network_params;
    
    public:
        mlp();
        ~mlp();
        network_params* params;
        network_params* init_network();
        double random_double();
};

mlp::mlp(){
    params = new network_params;
}

mlp::~mlp(){
    delete params;
}


double mlp::random_double(){
    srand(time(0));
    double rnum = (double)rand() / RAND_MAX;
    return rnum; // return a random double value between 0 and 1
}

mlp::network_params* mlp::init_network(){
    /**
     * To write an init function first we need to allocate memory for the weights using malloc.
    */

    params->num_weights = INPUT_DIM + HIDDEN_LAYER_SIZE*NUM_HIDDEN_LAYERS + OUTPUT_DIM;
    
    params->weights = static_cast<double*>(malloc(sizeof(double) * params->num_weights));
    params->biases = static_cast<double*>(malloc(sizeof(double) * params->num_weights));
    params->activations = static_cast<double*>(malloc(sizeof(double) * params->num_weights));
    
    int count = 0;

    for(int i = 0; i < params->num_weights ; i++){
        params->weights[i] = random_double();
        params->biases[i] = random_double();
        params->activations[i] = 0.00;
        // printf("Initialized neuron : %d \n", i);
        count++;
    }


    if(count == params->num_weights){
        printf("Successfully initialized all neurons. \n");
        printf("Total neurons initialized : %d \n", count);
    printf("Total number of neurons should be : %d \n", params->num_weights);
    }

   return params;
}

int main(){
    mlp *nn = new mlp();
    nn->init_network();
}
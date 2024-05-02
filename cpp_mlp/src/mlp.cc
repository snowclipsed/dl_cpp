// #include "data_handler.hpp"
// #include "data.hpp"
#include <cstdlib>
#include <vector>
#include "stdio.h"
#include "stdint.h"
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include "loguru.cpp"
#include "loguru.hpp"
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
        double sigmoid_activation(double x);
        double relu_activation(double x);
        void forward_pass();
        std::vector<double> mat_mul(std::vector<double> in, std::vector<double> weights, std::vector<double> biases, std::vector<double>out, int activation);
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
    return rnum; // return a randoconst std::string& activationm double value between 0 and 1
}

mlp::network_params* mlp::init_network(){
    /**
     * To write an init function first we need to allocate memory for the weights using malloc.
    */

    params->num_weights = INPUT_DIM * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE*HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE*OUTPUT_DIM;
    
    params->weights = static_cast<double*>(malloc(sizeof(double) * params->num_weights));
    params->biases = static_cast<double*>(malloc(sizeof(double) * params->num_weights));
    params->activations = static_cast<double*>(malloc(sizeof(double) * params->num_weights));
    float total_mem = sizeof(params->num_weights) + sizeof(params->weights) + sizeof(params->biases) + sizeof(params->activations);
    int count = 0;

    for(int i = 0; i < params->num_weights ; i++){
        params->weights[i] = random_double();
        params->biases[i] = random_double();
        params->activations[i] = 0.00;
        // RAW_LOG_F(0, "Initializing Neuron");
        count++;
    }


    if(count == params->num_weights){
        LOG_F(0, "Successfully initialized all weights with random values.");
        LOG_F(0, "Total weights initialized : %d", count);
        LOG_F(0, "Total number of weights should be : %d", params->num_weights);
        LOG_F(0, "Total memory allocated for weights and biases is %f Bytes", total_mem);
    }

   return params;
}


double mlp::sigmoid_activation(double x){

    x = 1 / (1 + exp(x * -1));
    // printf("activation: %f", x);
    return x;
}

double mlp::relu_activation(double x){
    return std::max(0.00, x);
}




std::vector<double> mlp::mat_mul(std::vector<double> X, std::vector<double> W, std::vector<double> B, std::vector<double> Z, int activation){
/**
 * X = Input matrix 
 * Z = Output matrix
 * W = Weight matrix (Size = X * Z) 784 * 256
 * B = Bias matrix 
 * forward pass eqn is => Z = W * X + B
 * 
 * X = [X1,X2,X3,X4,X5]
 * Z = [A1,A2,A3,A4]
 * W = [W1,W2,W3,W4,W5...W20]
 * B = [B1, B2, B3, B4]
 * 
 * activation:
 *  0 = no activation
 *  1 = sigmoid
 *  2 = ReLU
 * 
 *
*/
    for (int i = 0; i<Z.size(); i++){
        double sum = 0.0;
        for(int j =0; j<X.size(); j++){
            sum += X[j]*W[X.size()*i+j];
        }
        switch(activation){

        case 1:
            Z[i] = sigmoid_activation(sum + B[i]);
        case 2:
            Z[i] = relu_activation(sum + B[i]);
        default:
            Z[i] = sigmoid_activation(sum + B[i]);
        }
    }
return Z;
}


void mlp::forward_pass(){


}


int main(){
    mlp *nn = new mlp();
    nn->init_network();

    std::vector<double> X = {1.0,2.0,3.0,4.0};
    std::vector<double> W = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0};
    std::vector<double> B = {0.0,0.0,0.0};
    std::vector<double> Z = {0.0,0.0,0.0};
    nn->mat_mul(X,W,B,Z,0);


}
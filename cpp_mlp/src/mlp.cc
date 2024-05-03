// #include "data_handler.hpp"
// #include "data.hpp"
#include "data_handler.hpp"
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
        std::vector<double*> weights;
        std::vector<double*> biases;
        std::vector<double*> activations;
        int num_activations;
        int num_weights;
        int num_biases;
    }network_params;
    
    public:
        mlp();
        ~mlp();
        network_params* params;
        network_params* init_network(network_params* params);
        double* random_double();
        double sigmoid_activation(double x);
        double relu_activation(double x);
        void forward_pass(network_params* params);
        std::vector<double> mat_mul(std::vector<double> in, std::vector<double> weights, std::vector<double> biases, std::vector<double>out, int activation);
};

mlp::mlp(){
    params = new network_params;
}

mlp::~mlp(){
    delete params;
}


double* mlp::random_double(){
    srand(time(0));
    double* rnum = new double; 
    *rnum = static_cast<double>(rand()) / RAND_MAX;
    return rnum; // return a randoconst std::string& activationm double value between 0 and 1
}

mlp::network_params* mlp::init_network(network_params* params){
    /**
     * To write an init function first we need to allocate memory for the weights using malloc.
    */

    params->num_weights = INPUT_DIM * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE*HIDDEN_LAYER_SIZE*(NUM_HIDDEN_LAYERS-1) + HIDDEN_LAYER_SIZE*OUTPUT_DIM;
    params->num_activations = HIDDEN_LAYER_SIZE*NUM_HIDDEN_LAYERS + OUTPUT_DIM;
    params->num_biases = HIDDEN_LAYER_SIZE*NUM_HIDDEN_LAYERS + OUTPUT_DIM;
    params->weights.resize(params->num_weights);
    params->biases.resize(params->num_biases);
    params->activations.resize(params->num_activations);
    LOG_F(0, "Length of activations vector %ld", params->activations.size());
    int count = 0;

    for(int i = 0; i < params->num_weights ; i++){
        params->weights[i] = random_double();
        count++;
    }
    LOG_F(0, "Successfully initialized all weights with random values.");

    for(int i = 0; i < params -> num_activations; i++){
        params->biases[i] = random_double();
        double* zero = new double; 
        *zero = 0.000;
        params->activations[i]= zero;   
    }
    LOG_F(0, "Successfully initialized all biases with random values.");
    LOG_F(0, "Successfully initialized all activations to zero.");

    if(count == params->num_weights){
        
        LOG_F(0, "Total weights initialized : %d", count);
        LOG_F(0, "Total number of weights should be : %d", params->num_weights);
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
            LOG_F(0, "Using sigmoid activation.");
        case 2:
            Z[i] = relu_activation(sum + B[i]);
            LOG_F(0, "Using ReLU activation.");
        default:
            Z[i] = sigmoid_activation(sum + B[i]);
            LOG_F(0, "Using sigmoid activation.");
        }
    }
return Z;
}


void mlp::forward_pass(network_params* params){
    /**
     * Forward pass takes input image vector and calculates the values for the activations for different layers.
     * First we calculate the weights and activations for the first hidden layer, then we use those activations for second hidden layer, 
     * and then the third activation for output layer
    */

   //Between input layer and first hidden layer

    // Z = first hidden, X = input, W = weights

    


    // for (int i = 0; i<NUM_HIDDEN_LAYERS; i++)

}


int main(){

    data_handler *dh = new data_handler();
    dh->load_feature_vectors("/home/snow/learn/dl_cpp/cpp_mlp/dataset/train-images-idx3-ubyte");
    dh->load_feature_labels("/home/snow/learn/dl_cpp/cpp_mlp/dataset/train-labels-idx1-ubyte");
    dh->split_data();
    dh->class_counter();

    mlp *nn = new mlp();
    nn->init_network(nn->params);

    // std::vector<double> X = {1.0,2.0,3.0,4.0};
    // std::vector<double> W = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0};
    // std::vector<double> B = {0.0,0.0,0.0};
    // std::vector<double> Z = {0.0,0.0,0.0};
    // nn->mat_mul(X,W,B,Z,0);


}
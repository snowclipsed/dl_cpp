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
        void forward_pass(network_params* params, std::vector<Data*> train);
        std::vector<double*> mat_mul(std::vector<double*> X, int X_start, int X_end, std::vector<double*> W, int W_start, int W_end, std::vector<double*> B, std::vector<double*> Z, int Z_start, int Z_end, int activation);
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




std::vector<double*> mlp::mat_mul(std::vector<double*> X, int X_start, int X_end, std::vector<double*> W, int W_start, int W_end, std::vector<double*> B, std::vector<double*> Z, int Z_start, int Z_end, int activation){
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


    for (int i = 0; i<Z_end-Z_start; i++){
        double sum = 0.0;
        for(int j = 0; j<X_end-X_start; j++){
            sum += *X[X_start+j] * *W[W_start+(X_end-X_start)*i+j];
            // offset of 784 * 256
        }
        switch(activation){

        case 1:
            *Z[Z_start+i] = sigmoid_activation(sum + *B[Z_start+i]);
            // LOG_F(0, "Z = %f", *Z[i]);
            // LOG_F(0, "Using sigmoid activation.");
            break;
        case 2:
            *Z[Z_start+i] = relu_activation(sum + *B[Z_start+i]);
            // LOG_F(0, "Z = %f", *Z[i]);
            // LOG_F(0, "Using ReLU activation.");
            break;
        default:
            *Z[Z_start+i] = sigmoid_activation(sum + *B[Z_start+i]);
            // LOG_F(0, "Z = %f", *Z[i]);
            // LOG_F(0, "Using sigmoid activation.");
            break;
        }
    }
return Z;
}


std::vector<double*> convertVector(const std::vector<uint8_t>* input) {
    // LOG_F(0, "Converting uint_8* vector to double* vector");
    std::vector<double*> output;
    output.reserve(input->size());  // Reserve memory for efficiency

    for (uint8_t value : *input) {
        double* ptr = new double(static_cast<double>(value));
        output.push_back(ptr);
    }
    // LOG_F(0, "Converted uint_8* vector to double* vector");
    return output;
}


void mlp::forward_pass(network_params* params, std::vector<Data*> train){
    /**
     * Forward pass takes input image vector and calculates the values for the activations for different layers.
     * First we calculate the weights and activations for the first hidden layer, then we use those activations for second hidden layer, 
     * and then the third activation for output layer
    */

   //Between input layer and first hidden layer

    // Z = first hidden, X = input, W = weights
    
    for(int i = 0; i<train.size(); i++){
        // initially we put in the input vector of size 784 into the first hidden layer of size 256.
        int X_start = 0;
        int X_end = INPUT_DIM;
        int Z_start = 0;
        int Z_end = HIDDEN_LAYER_SIZE;
        int W_start = 0;
        int W_end = INPUT_DIM * HIDDEN_LAYER_SIZE;
        // LOG_F(0, "%d, %d, %d, %d, %d, %d", X_start, X_end, Z_start, Z_end, W_start, W_end);
        mat_mul(convertVector(train[i]->get_features()), 0, INPUT_DIM, params->weights, X_start, X_end, params->biases, params->activations, Z_start, Z_end, 2);
        // LOG_F(0, "Input layer for image number : %d", i);

        

        for (int layer = 0; layer<NUM_HIDDEN_LAYERS-1; layer++){
           X_start = Z_start;
           X_end = Z_end;
           Z_start = Z_end;
           Z_end += HIDDEN_LAYER_SIZE;
           W_start = W_end;
           W_end += HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE; 

            // LOG_F(0, "%d, %d, %d, %d, %d, %d", X_start, X_end, Z_start, Z_end, W_start, W_end);
            mat_mul(params->activations, X_start, X_end, params->weights, W_start, W_end, params->biases, params->activations, Z_start, Z_end, 2);
            // LOG_F(0, "Hidden layer %d for image number : %d", layer+1, i);
        }

            X_start = HIDDEN_LAYER_SIZE * (NUM_HIDDEN_LAYERS-1);
            X_end = HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS;
            Z_start = HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS;
            Z_end = HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS + OUTPUT_DIM;
            W_start = INPUT_DIM * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE*HIDDEN_LAYER_SIZE*(NUM_HIDDEN_LAYERS-1);
            W_end = INPUT_DIM * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE*HIDDEN_LAYER_SIZE*(NUM_HIDDEN_LAYERS-1) + HIDDEN_LAYER_SIZE*OUTPUT_DIM;
            // LOG_F(0, "%d, %d, %d, %d, %d, %d", X_start, X_end, Z_start, Z_end, W_start, W_end);
            mat_mul(params->activations, X_start, X_end, params->weights, W_start, W_end, params->biases, params->activations, Z_start, Z_end, 1);

    LOG_F(0, "Forward pass for image : %d", i);
    }


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
    nn->forward_pass(nn->params, dh->get_train());

}
#include "data_handler.hpp"
#include <cstdlib>
#include <vector>
#include "stdio.h"
#include "stdint.h"
#include <cmath>
#include <ctime>
#include <string>
#include <random>
#include <algorithm>
#include "loguru.cpp"
#include "loguru.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "implot_internal.h"
#include <stdio.h>
#include <future>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

// This example can also compile and run with Emscripten! See 'Makefile.emscripten' for details.
#ifdef __EMSCRIPTEN__
#include "../libs/emscripten/emscripten_mainloop_stub.h"
#endif




#define INPUT_DIM 784
#define OUTPUT_DIM 10
#define HIDDEN_LAYER_SIZE 256
#define NUM_HIDDEN_LAYERS 3
#define BATCH_SIZE 1
#define LEARNING_RATE 1e-5
#define NUM_EPOCHS 3
#define EPSILON 1e-5


class mlp{

    typedef struct{
        std::vector<double*> features;
        std::vector<double*> weights;
        std::vector<double*> weight_gradients; // weight gradients
        std::vector<double*> bias_gradients;
        std::vector<double*> biases;
        std::vector<double*> logits;
        std::vector<double*> gamma; //batchnorm learnable
        std::vector<double*> beta; //batchnorm learnable
        std::vector<double*> norm_logits;
        std::vector<double*> scaled_logits;
        std::vector<double*> activations;
        std::vector<double*> error_term;
        std::vector<double*> pred;
        std::vector<double*> true_pred;
        int num_activations;
        int num_weights;
        int num_biases;
    }network_params;
    
    public:
        mlp();
        ~mlp();
        
        network_params* params;
        network_params* init_network(network_params* params, std::vector<Data*> train);


        const int input_weights_size = INPUT_DIM * HIDDEN_LAYER_SIZE;
        const int hidden_weights_size = HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE;
        const int output_weights_size = HIDDEN_LAYER_SIZE * OUTPUT_DIM;


        
        double* random_double();
        double sigmoid_activation(double x);
        double relu_activation(double x);
        double sigmoid_der(double x);
        double relu_der(double x);
        
        void forward_pass(network_params* params, std::vector<Data*> batch);
        void backward_pass(network_params* params, std::vector<Data*> batch);
        
        void create_one_hot(network_params* params, std::vector<Data*> train, int num_classes);
        std::vector<double*> batch_norm(std::vector<double*> logits, std::vector<double*> norm_logits, int logit_start, int logit_end, double epsilon);
        std::vector<Data*> create_batch(std::vector<Data*> data);
        std::vector<double*> mat_mul(network_params* params, std::vector<double*> input, int X_start, int X_end, int W_start, int W_end, int Z_start, int Z_end, bool is_input, int activation, bool batchnnorm);
        std::vector<double*> mat_mul_error_term(network_params* params, int activation_offset, int weight_offset, int previous_error_offset, int dim1, int dim2);
        std::vector<double*> mat_mul_error_term(network_params* params, int activation_offset, int weight_offset, int dim1);
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

mlp::network_params* mlp::init_network(network_params* params, std::vector<Data*> train){
    /**
     * To write an init function first we need to allocate memory for the weights using malloc.
    */

    for(int i = 0; i<train.size(); i++){
        convertVector(train[i]->get_features(), params->features);
    }

    params->norm_logits.resize(params->num_activations);
    params->scaled_logits.resize(params->num_activations);
    params->activations.resize(params->num_activations);
    params->error_term.resize(params->num_activations);
    params->weight_gradients.resize(params->num_weights);
    params->bias_gradients.resize(params->num_activations);
    LOG_F(0, "Length of activations vector %ld", params->activations.size());
    int count = 0;

    for(int i = 0; i < params->num_weights ; i++){
        params->weights[i] = random_double();
        params->weight_gradients[i] = random_double();
        count++;
    }
    LOG_F(0, "Successfully initialized all weights with random values.");

    double* zero = new double; 
    *zero = 0.000;

    for(int i = 0; i < params -> num_activations; i++){
        params->biases[i] = random_double();
        params->bias_gradients[i] = random_double();
        params->activations[i]= zero;  
        params->logits[i] = zero;
        params->norm_logits[i] = zero;
        params->scaled_logits[i] = zero;
        params->gamma[i] = random_double();
        params->beta[i] = random_double();
        params->error_term[i] = zero;
    }
    LOG_F(0, "Successfully initialized all biases with random values.");
    LOG_F(0, "Successfully initialized all activations to zero.");

    if(count == params->num_weights){
        
        LOG_F(0, "Total weights initialized : %d", count);
        LOG_F(0, "Total number of weights should be : %d", params->num_weights);
    }

   return params;
}

/**
 * Applies the sigmoid activation function to the input value.
 *
 * The sigmoid function maps any real-valued number to a value between 0 and 1.
 * This is often used as an output layer activation function in neural networks.
 *
 * @param x The input value to be activated.
 * @return The result of applying the sigmoid function to the input value.
 */
double mlp::sigmoid_activation(double x){

    x = 1 / (1 + exp(x * -1));
    // printf("activation: %f", x);
    return x;
}

double mlp::relu_activation(double x){
    return std::max(0.00, x);
}


std::vector<double*> mlp::mat_mul(network_params* params, std::vector<double*> input, int X_start, int X_end, int W_start, int W_end, int Z_start, int Z_end, bool is_input, int activation, bool batchnnorm){

    if(is_input){

            for (int i = 0; i<Z_end-Z_start; i++){
            double sum = 0.0;
            for(int j = 0; j<X_end-X_start; j++){
                sum += *input[X_start+j] * *params->weights[W_start+(X_end-X_start)*i+j];
            }
            *params->logits[Z_start+i] = sum + *params->biases[Z_start+i];
        }

    }else{

        for (int i = 0; i<Z_end-Z_start; i++){
            double sum = 0.0;
            for(int j = 0; j<X_end-X_start; j++){
                sum += *params->activations[X_start+j] * *params->weights[W_start+(X_end-X_start)*i+j];
            }
            *params->logits[Z_start+i] = sum + *params->biases[Z_start+i];
        }
    }

    if(batchnnorm){
        batch_norm(params->logits, params->norm_logits, Z_start, Z_end, EPSILON);
    }

    for (int i = 0; i<Z_end-Z_start; i++){
        *params->scaled_logits[Z_start + i] = *params->norm_logits[Z_start+i]* *params->gamma[Z_start + i] + *params->beta[Z_start+i];
        

        switch(activation){

        case 1:
            *params->activations[Z_start+i] = sigmoid_activation(*params->scaled_logits[Z_start+i]); //scaled activation
            // LOG_F(0, "Using sigmoid activation.");
            break;
        case 2:
            *params->activations[Z_start+i] = relu_activation(*params->scaled_logits[Z_start+i]);
            // LOG_F(0, "Using ReLU activation.");
            break;
        default:
            *params->activations[Z_start+i] = sigmoid_activation(*params->scaled_logits[Z_start+i]);
            // LOG_F(0, "Using sigmoid activation.");
            break;
        }
    }
return params->activations;
}



// std::vector<double*> convertVector(const std::vector<uint8_t>* input, std::vector<double*> output) {
//     // LOG_F(0, "Converting uint_8* vector to double* vector");
    
//         for (uint8_t value : *input) {
//             double* ptr = new double(static_cast<double>(value));
//             output.push_back(ptr);
//         }
//         // LOG_F(0, "Converted uint_8* vector to double* vector");
//         return output;
//     }

void convertVector(const std::vector<uint8_t>* inputVector, std::vector<double*>& outputVector) {
    // Check if the input vector pointer is not null
    if (inputVector) {
        // Clear the output vector to avoid appending to existing elements
        outputVector.clear();

        // Reserve enough space in the output vector to improve performance
        outputVector.reserve(inputVector->size());

        // Iterate over the input vector and convert each uint8_t value to a double pointer
        for (uint8_t value : *inputVector) {
            double* newDouble = new double(static_cast<double>(value));
            outputVector.push_back(newDouble);
            delete newDouble;
        }
    } else {
        // Handle the case where the input vector pointer is null
        outputVector.clear();
    }
}

void convertFeatures(const std::vector<uint8_t>* inputVector, std::vector<double*>& outputVector) {
    // Check if the input vector pointer is not null
    if (inputVector) {
        // Clear the output vector to avoid appending to existing elements
        outputVector.clear();

        // Reserve enough space in the output vector to improve performance
        outputVector.reserve(inputVector->size());

        // Iterate over the input vector and convert each uint8_t value to a double pointer
        for (uint8_t value : *inputVector) {
            double* newDouble = new double(static_cast<double>(value));
            outputVector.push_back(newDouble);
            delete newDouble;
        }
    } else {
        // Handle the case where the input vector pointer is null
        outputVector.clear();
    }
}


/**
 * @brief Creates a batch of data by randomly selecting examples from the input data.
 *
 * This function creates a batch of data by randomly shuffling the input data vector
 * and selecting the first `BATCH_SIZE` elements from the shuffled vector. The batch
 * is used for training or evaluation of the neural network.
 *
 * @param data A vector of pointers to Data objects representing the input data.
 *
 * @return A vector of pointers to Data objects representing the created batch.
 *
 * @note The function assumes that the input data vector has at least `BATCH_SIZE`
 *       elements. If the input data vector has fewer elements than `BATCH_SIZE`,
 *       the function will return a batch containing all available elements.
 *       The constant `BATCH_SIZE` should be defined elsewhere in the code.
 */
std::vector<Data*> mlp::create_batch(std::vector<Data*> data){
    std::vector<Data*> batch;
    batch.reserve(BATCH_SIZE);

    // Create a random number generator and shuffler
    std::random_device rd;
    std::mt19937 gen(rd());

    // Shuffle the input data vector
    std::vector<Data*> shuffled_data = data;
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), gen);

    // Select the first batch_size elements from the shuffled vector
    for (int i = 0; i < BATCH_SIZE; ++i) {
        batch.push_back(shuffled_data[i]);
    }
    return batch;
}



std::vector<double*> mlp::batch_norm(std::vector<double*> logits, std::vector<double*> norm_logits, int logit_start, int logit_end, double epsilon) {
    // Compute the mean of the activations
    double sum = 0.0;
    for (int i = 0; i<logit_end-logit_start; i++) {
        sum += *logits[logit_start + i];
    }
    double mean = sum / logits.size();
    
    // Compute the variance of the activations
    double variance = 0.0;
    for (int i = 0; i < logit_end-logit_start; i++) {
        double logit = *logits[logit_start + i];
        variance += (logit - mean) * (logit - mean);
    }
    variance /= logits.size();

    for (int i = 0; i < logit_end-logit_start; i++) {
        *norm_logits[logit_start + i] = (*logits[i] - mean) / std::sqrt(variance + epsilon);
    }

    return norm_logits;
}


/**
 * @brief Performs the forward pass of the mlp.
 *
 * This function takes the input batch of data and propagates it through the MLP
 * by performing matrix multiplications and applying activation functions at each layer.
 * The computed activations are stored in the network_params object.
 *
 * @param params A pointer to the network_params object containing the weights, biases,
 *               and activations for the network.
 * 
 * @param batch A vector of pointers to Data objects, representing the input batch.
 *
 */
void mlp::forward_pass(network_params* params, std::vector<Data*> batch){


    //Between input layer and first hidden layer

    // Z = first hidden, X = input, W = weights
    // LOG_F(0, "Initializing forward pass.");
    std::vector<double*> batch_input;
    
        // initially we put in the input vector of size 784 into the first hidden layer of size 256.
        int X_start = 0;
        int X_end = INPUT_DIM;
        int Z_start = 0;
        int Z_end = HIDDEN_LAYER_SIZE;
        int W_start = 0;
        int W_end = INPUT_DIM * HIDDEN_LAYER_SIZE ;
    for(unsigned long i = 0; i<batch.size(); i++){    
        convertVector(batch[i]->get_features(), batch_input);
        // LOG_F(0, "%d, %d, %d, %d, %d, %d", X_start, X_end, Z_start, Z_end, W_start, W_end);
        // network_params* params, std::vector<double*> input, int X_start, int X_end, int W_start, int W_end, int Z_start, int Z_end, bool is_input, int activation, bool batchnnorm
        mat_mul(params, batch_input, 0, INPUT_DIM, X_start, X_end, Z_start, Z_end, true, 2, true);
        // LOG_F(0, "Input layer for image number : %d", i);
    }
        

    for (int layer = 0; layer<NUM_HIDDEN_LAYERS-1; layer++){
        for(unsigned long i = 0; i<batch.size(); i++){
           X_start = Z_start;
           X_end = Z_end;
           Z_start = Z_end;
           Z_end += HIDDEN_LAYER_SIZE;
           W_start = W_end;
           W_end += HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE; 

            // LOG_F(0, "%d, %d, %d, %d, %d, %d", X_start, X_end, Z_start, Z_end, W_start, W_end);
            mat_mul(params, batch_input, X_start, X_end, W_start, W_end, Z_start, Z_end, false, 2, true);
            // LOG_F(0, "Hidden layer %d for image number : %d", layer+1, i);
        }
    }
            X_start = HIDDEN_LAYER_SIZE * (NUM_HIDDEN_LAYERS-1);
            X_end = HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS;
            Z_start = HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS;
            Z_end = HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS + OUTPUT_DIM;
            W_start = INPUT_DIM * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE*HIDDEN_LAYER_SIZE*(NUM_HIDDEN_LAYERS-1);
            W_end = INPUT_DIM * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE*HIDDEN_LAYER_SIZE*(NUM_HIDDEN_LAYERS-1) + HIDDEN_LAYER_SIZE*OUTPUT_DIM;
            // LOG_F(0, "%d, %d, %d, %d, %d, %d", X_start, X_end, Z_start, Z_end, W_start, W_end);

        for(unsigned long i = 0; i<batch.size(); i++){
            mat_mul(params, batch_input, X_start, X_end, W_start, W_end, Z_start, Z_end, false, 1, true);
            params->pred.insert(params->pred.end(), params->activations.end()-10, params->activations.end());
        
            // LOG_F(0, "Forward pass for image : %d", i);
    }
}


/**
 * @brief Creates a one-hot vector representation of the class labels for the given training data.
 *
 * This function iterates over the training data and constructs a one-hot vector representation
 * of the class labels. For each data instance, a vector of length `num_classes` is created,
 * where all elements are set to 0, except for the element corresponding to the class label,
 * which is set to 1.
 *
 * The one-hot vectors are stored in the `true_pred` vector of the `network_params` object,
 * where each element is a pointer to a `double` value representing the corresponding element
 * in the one-hot vector.
 *
 * @param params Pointer to the `network_params` object containing the `true_pred` vector.
 * @param train Vector of pointers to `Data` objects representing the training data.
 * @param num_classes Number of classes in the training data.
 *
 * @throw std::exception (or any derived exception) if any error occurs during the operation.
 *
 * @note The function dynamically allocates memory for the `double` pointers stored in the
 *       `true_pred` vector. It is the responsibility of the caller to ensure proper memory
 *       management and deallocation of these pointers when they are no longer needed.
 */
void mlp::create_one_hot(network_params* params, std::vector<Data*> train, int num_classes){
    /**
     * 
    */

    // LOG_F(0, "Creating one hot vector.");
    

    for(long unsigned image = 0; image<train.size(); image++){
        for(int classlabel = 0 ; classlabel < num_classes; classlabel++){
            if (train[image]->get_enumerated_class_label() == classlabel){
                double labelvalue = 1.00;
                double* label_ptr = new double(labelvalue);
                params->true_pred.push_back(label_ptr);
            }else{
                double labelvalue = 0.00;
                double* label_ptr = new double(labelvalue);
                params->true_pred.push_back(label_ptr);
            }
        }
        // LOG_F(0, "One hot vector created for image %d", image);
    }
}

/**
 * @brief Calculates the cross-entropy loss for multi-class classification.
 *
 * The cross-entropy loss is a commonly used loss function for multi-class
 * classification tasks. It measures the performance of a model by comparing
 * the predicted probability distribution over classes with the true distribution.
 *
 * @param y_true A vector representing the one-hot encoded true class labels.
 *               The vector should contain 0s and a single 1, where the index
 *               of the 1 corresponds to the true class label.
 * @param y_pred A vector representing the predicted probability distribution
 *               over classes. The sum of the elements in this vector should be 1.
 *
 * @return The cross-entropy loss as a scalar value.
 *
 * @note The input vectors y_true and y_pred should have the same size (number of classes).
 *       The y_pred vector should contain valid probability values (non-negative and summing to 1).
 */
double cross_entropy_loss(std::vector<double*> pred, std::vector<double*> true_pred) {
    double loss = 0.0;
    const int num_classes = 10; // Number of classes for MNIST (0-9)

    for (long unsigned i = 0; i < true_pred.size(); i++) {
        double sum_log_probs = 0.0;
        for (int j = 0; j < num_classes; j++) {
            double pred_clipped = std::max(pred[i][j], 1e-15);
            sum_log_probs += true_pred[i][j] * log2(pred_clipped);
        }
        loss += -sum_log_probs;
    }

    return loss / true_pred.size();
}

double mlp::sigmoid_der(double x){
    return x * (1.0 - x);
}


double mlp::relu_der(double x) {
    return x > 0 ? 1 : 0;
}


std::vector<double*> mlp::mat_mul_error_term(network_params* params, int activation_offset, int weight_offset, int dim1){
    
    
    int y_size = params->true_pred.size();
        
    for(int i = 0; i<OUTPUT_DIM; i++){
        *params->error_term[activation_offset + i] = *params->pred[y_size - OUTPUT_DIM + i] - *params->true_pred[y_size - OUTPUT_DIM + i];
    }      

    return params->error_term;
    }

std::vector<double*> mlp::mat_mul_error_term(network_params* params, int activation_offset, int weight_offset, int previous_error_offset, int dim1, int dim2){
    for(int i = 0; i<dim2; i++){
        for(int j = 0; j<dim1; j++){
            *params->error_term[activation_offset + i] +=  *params->weights[weight_offset + HIDDEN_LAYER_SIZE*j + i] * *params->error_term[previous_error_offset + j];
            *params->error_term[activation_offset + i] = *params->error_term[activation_offset + i] * relu_der(*params->activations[activation_offset + i]);
        }
    }
    return params->error_term;
}






void mlp::backward_pass(network_params* params, std::vector<Data*> batch){

    
    
    std::vector<double*> features;
    features.resize(INPUT_DIM);

    int i = 0;
    int j = 0;
    int layer = 0;

    int activation_offset = params->num_activations - OUTPUT_DIM;
    int weight_offset;
    int previous_error_offset = 0;

    

    for(long unsigned image = 0; image<batch.size(); image++){
        
        convertVector(batch[image]->get_features(), features);


        mat_mul_error_term(params, activation_offset, weight_offset, OUTPUT_DIM);

        //calculate the error term for last hidden layer
        previous_error_offset = activation_offset;
        activation_offset = params->num_activations-(OUTPUT_DIM+HIDDEN_LAYER_SIZE);
        weight_offset = params->num_weights-(output_weights_size);
        mat_mul_error_term(params, activation_offset, weight_offset, previous_error_offset, OUTPUT_DIM, HIDDEN_LAYER_SIZE);


        //calculate the error term for the other hidden layers
        for(layer = 0; layer < NUM_HIDDEN_LAYERS-1; layer++){
            previous_error_offset = activation_offset;
            activation_offset -= HIDDEN_LAYER_SIZE;
            weight_offset -= hidden_weights_size;
            mat_mul_error_term(params, activation_offset, weight_offset, previous_error_offset, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);
        }

        /*
        Calculating gradients.
        */

        // calculate weight gradient for output neurons
        // del = grad * logits of previous layer 
        activation_offset = params->num_activations - OUTPUT_DIM;
        weight_offset = params->num_weights - output_weights_size;
        int logit_offset = params->num_activations - (OUTPUT_DIM + HIDDEN_LAYER_SIZE);

        for(i = 0; i<OUTPUT_DIM; i++){
            for(j = 0; j< HIDDEN_LAYER_SIZE; j++){
                *params->weight_gradients[weight_offset + i * HIDDEN_LAYER_SIZE + j] += *params->error_term[activation_offset + i] * *params->logits[logit_offset + j];
            }
        }

        //bias gradient calculation
        for(i = 0; i<OUTPUT_DIM; i++){
            *params->bias_gradients[activation_offset + i] = *params->error_term[activation_offset + i];
        }
                
        // calculate the gradients of the hidden weights
        for(layer = 0; layer<NUM_HIDDEN_LAYERS-1; layer++){
            activation_offset -= HIDDEN_LAYER_SIZE;
            weight_offset -= hidden_weights_size;
            logit_offset -= HIDDEN_LAYER_SIZE;
            for(i = 0; i<HIDDEN_LAYER_SIZE; i++){
                for(j=0; j<HIDDEN_LAYER_SIZE; j++){
                    *params->weight_gradients[weight_offset + i * HIDDEN_LAYER_SIZE + j] += *params->error_term[activation_offset + i] * *params->logits[logit_offset + j];
                }
            }

            for(i = 0; i<HIDDEN_LAYER_SIZE; i++){
                *params->bias_gradients[activation_offset + i] = *params->error_term[activation_offset + i];
            }
        }
        

        for(i = 0; i<HIDDEN_LAYER_SIZE; i++){
                for(j=0; j<INPUT_DIM; j++){
                    *params->weight_gradients[weight_offset + i * HIDDEN_LAYER_SIZE + j] += *params->error_term[activation_offset + i] * *features[j];
                }
            }

        for(i = 0; i<HIDDEN_LAYER_SIZE; i++){
            *params->bias_gradients[activation_offset + i] = *params->error_term[activation_offset + i];
        }
    }   

    weight_offset = params->num_weights - output_weights_size;
    int index = 0;
    for(i = 0; i<OUTPUT_DIM; i++){
        for(j = 0; j<HIDDEN_LAYER_SIZE; j++){
            index = i*HIDDEN_LAYER_SIZE + j;
            *params->weights[weight_offset + index] -= LEARNING_RATE * *params->weight_gradients[weight_offset + index];
        }
    }

    int bias_offset = params->num_biases - OUTPUT_DIM;
    for(i = 0; i<OUTPUT_DIM; i++){
        *params->bias_gradients[bias_offset + i] -= LEARNING_RATE * *params->bias_gradients[bias_offset + i];
    }

    for(layer = 0; layer<NUM_HIDDEN_LAYERS-1; layer++){
        weight_offset -= hidden_weights_size;
        bias_offset -= HIDDEN_LAYER_SIZE;
        // LOG_F(0, "weight_offset: %d, bias_offset: %d", weight_offset, bias_offset);
        for(i = 0; i<HIDDEN_LAYER_SIZE; i++){
            for(j = 0; j<HIDDEN_LAYER_SIZE; j++){
                index = i*HIDDEN_LAYER_SIZE + j;
                *params->weights[weight_offset + index] -= LEARNING_RATE * *params->weight_gradients[weight_offset + index];
            }
        }

        for(i = 0; i<HIDDEN_LAYER_SIZE; i++){
            *params->bias_gradients[bias_offset + i] -= LEARNING_RATE * *params->bias_gradients[bias_offset + i];
        }
    }

    weight_offset -= input_weights_size;
    // LOG_F(0, "weight_offset: %d, bias_offset: %d", weight_offset, bias_offset);
    for(i = 0; i<HIDDEN_LAYER_SIZE; i++){
        for(j = 0; j<INPUT_DIM; j++){
            index = i*INPUT_DIM + j;
            *params->weights[weight_offset + index] -= LEARNING_RATE * *params->weight_gradients[weight_offset + index];
        }
    }
}

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}


double run_mlp(data_handler* dh, mlp* nn, int num_classes){

std::vector<Data*> batch = nn->create_batch(dh->get_train());

nn->create_one_hot(nn->params, batch, num_classes);
nn->forward_pass(nn->params, batch);
nn->backward_pass(nn->params, batch);
double CE_loss = cross_entropy_loss(nn->params->pred, nn->params->true_pred);
return CE_loss;
}

void train_network(data_handler* dh, mlp* nn, int num_classes, std::vector<double>& losses) {
    int num_batches = dh->get_train().size() / BATCH_SIZE;
    for(int epoch_num = 0; epoch_num < NUM_EPOCHS ; epoch_num++){
        for (int batch_num = 0; batch_num < num_batches; batch_num++) {
        double loss = run_mlp(dh, nn, num_classes);
        losses.push_back(loss);
        }
    }
}




int main(int, char**) {
    glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            return 1;
    // Setup GLFW window
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Neural Network Loss Visualization", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // // Initialize OpenGL loader
    // if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { return -1; }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Load Neural Network data
    data_handler* dh = new data_handler();
    dh->load_feature_vectors("/home/snow/learn/dl_cpp/cpp_mlp/dataset/train-images-idx3-ubyte");
    dh->load_feature_labels("/home/snow/learn/dl_cpp/cpp_mlp/dataset/train-labels-idx1-ubyte");
    dh->split_data();
    int num_classes = dh->class_counter();

    mlp* nn = new mlp();
    nn->init_network(nn->params, dh->get_train());

    std::vector<double> losses;
    std::thread training_thread(train_network, dh, nn, num_classes, std::ref(losses));

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Poll and handle events
        glfwPollEvents();

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Plotting
        ImGui::SetNextWindowSize(ImVec2(800,400), ImGuiCond_FirstUseEver);
        ImGui::Begin("Loss Plot");
        int plot_size = losses.size();
        float x_min = plot_size > 100 ? plot_size - 100 : 0; // Show last 100 points or all points if less
        float x_max = plot_size > 0 ? plot_size : 100; // Set x_max to number of points or 100 if no points

        // Calculate the y-axis range based on the data in losses

        double y_min = 0.00;
        double y_max = 1.00;

        if(!losses.empty()){
            y_min = *std::min_element(losses.begin(), losses.end());
            y_max = *std::max_element(losses.begin(), losses.end());
        }

        if (ImPlot::BeginPlot("Loss Curve")) {
            ImPlot::SetupAxes("Step","Loss");
            ImPlot::SetupAxesLimits(x_min, x_max, y_min, y_max, ImGuiCond_Always); // Set x and y limits to automatically fit data
            ImPlot::PlotLine("Loss", losses.data(), losses.size());

            // If there are any losses, plot a marker at the last point
            if (!losses.empty()) {
                double last_loss = losses.back();
                // Display the latest loss under the plot
                ImGui::Text("Current Loss: %.6f", last_loss);
            } else {
                ImGui::Text("Current Loss: N/A");
            }

            ImPlot::EndPlot();
        }

        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // RGBA: Black
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Wait for the training thread to finish
    training_thread.join();

    // Cleanup
    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    delete dh;
    delete nn;

    return 0;
}

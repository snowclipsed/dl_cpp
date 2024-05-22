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
#include <random>
#include <algorithm>
#include "loguru.cpp"
#include "loguru.hpp"

#define INPUT_DIM 784
#define OUTPUT_DIM 10
#define HIDDEN_LAYER_SIZE 256
#define NUM_HIDDEN_LAYERS 3
#define BATCH_SIZE 1
#define LEARNING_RATE 1e-3


class mlp{

    typedef struct{
        std::vector<double*> weights;
        std::vector<double*> weight_gradients; // weight gradients
        std::vector<double*> bias_gradients;
        std::vector<double*> biases;
        std::vector<double*> logits;
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
        std::vector<Data*> create_batch(std::vector<Data*> data);
        std::vector<double*> mat_mul(std::vector<double*> X, int X_start, int X_end, std::vector<double*> W, int W_start, int W_end, std::vector<double*> B, std::vector<double*> Z, int Z_start, int Z_end, std::vector<double*> A, int activation);
        std::vector<double*> backwards_mat_mul(std::vector<double*> mat_A, std::vector<double*> mat_B, std::vector<double*> gradients);
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

    params->num_weights = INPUT_DIM * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE*HIDDEN_LAYER_SIZE*(NUM_HIDDEN_LAYERS-1) + HIDDEN_LAYER_SIZE*OUTPUT_DIM;
    params->num_activations = HIDDEN_LAYER_SIZE*NUM_HIDDEN_LAYERS + OUTPUT_DIM;
    params->num_biases = HIDDEN_LAYER_SIZE*NUM_HIDDEN_LAYERS + OUTPUT_DIM;
    params->weights.resize(params->num_weights);
    params->biases.resize(params->num_biases);
    params->logits.resize(params->num_activations);
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

    for(int i = 0; i < params -> num_activations; i++){
        params->biases[i] = random_double();
        params->bias_gradients[i] = random_double();
        double* zero = new double; 
        *zero = 0.000;
        params->activations[i]= zero;  
        params->logits[i] = zero; 
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



/**
 * @brief Performs matrix multiplication and applies activation function.
 *
 * This function performs the matrix multiplication between the input matrix `X` and
 * the weight matrix `W`, and adds the bias matrix `B`. It then applies the specified
 * activation function to the result and stores it in the output matrix `Z`.
 * 
 * Forward pass eqn is => Z = W * X + B.
 * 
 * @param X A vector of pointers to double values representing the input matrix.
 * @param X_start The starting index of the input matrix `X` in the vector.
 * @param X_end The ending index (exclusive) of the input matrix `X` in the vector.
 * @param W A vector of pointers to double values representing the weight matrix. (Size = X * Z) 784 * 256.
 * @param W_start The starting index of the weight matrix `W` in the vector.
 * @param W_end The ending index (exclusive) of the weight matrix `W` in the vector.
 * @param B A vector of pointers to double values representing the bias matrix.
 * @param Z A vector of pointers to double values representing the output matrix.
 * @param Z_start The starting index of the output matrix `Z` in the vector.
 * @param Z_end The ending index (exclusive) of the output matrix `Z` in the vector.
 * @param activation An integer specifying the activation function to apply:
 *                   0 = no activation, 1 = sigmoid, 2 = ReLU.
 *
 * @return A vector of pointers to double values representing the output matrix `Z`.
 *
 * @note The function assumes that the input matrices `X` and `W` have compatible
 *       dimensions for matrix multiplication, and that the output matrix `Z` has
 *       the correct size to store the result.
 */
std::vector<double*> mlp::mat_mul(std::vector<double*> X, int X_start, int X_end, std::vector<double*> W, int W_start, int W_end, std::vector<double*> B, std::vector<double*> Z, int Z_start, int Z_end, std::vector<double*> A, int activation){


    for (int i = 0; i<Z_end-Z_start; i++){
        double sum = 0.0;
        for(int j = 0; j<X_end-X_start; j++){
            sum += *X[X_start+j] * *W[W_start+(X_end-X_start)*i+j];
            // offset of 784 * 256
        }
        *A[Z_start+i] = sum + *B[Z_start+i];
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
    for(int i = 0; i<batch.size(); i++){
        // initially we put in the input vector of size 784 into the first hidden layer of size 256.
        int X_start = 0;
        int X_end = INPUT_DIM;
        int Z_start = 0;
        int Z_end = HIDDEN_LAYER_SIZE;
        int W_start = 0;
        int W_end = INPUT_DIM * HIDDEN_LAYER_SIZE ;
        
        convertVector(batch[i]->get_features(), batch_input);
        // LOG_F(0, "%d, %d, %d, %d, %d, %d", X_start, X_end, Z_start, Z_end, W_start, W_end);
        mat_mul(batch_input, 0, INPUT_DIM, params->weights, X_start, X_end, params->biases, params->activations, Z_start, Z_end, params->logits, 2);
        // LOG_F(0, "Input layer for image number : %d", i);

        

        for (int layer = 0; layer<NUM_HIDDEN_LAYERS-1; layer++){
           X_start = Z_start;
           X_end = Z_end;
           Z_start = Z_end;
           Z_end += HIDDEN_LAYER_SIZE;
           W_start = W_end;
           W_end += HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE; 

            // LOG_F(0, "%d, %d, %d, %d, %d, %d", X_start, X_end, Z_start, Z_end, W_start, W_end);
            mat_mul(params->activations, X_start, X_end, params->weights, W_start, W_end, params->biases, params->activations, Z_start, Z_end, params->logits , 2);
            // LOG_F(0, "Hidden layer %d for image number : %d", layer+1, i);
        }

            X_start = HIDDEN_LAYER_SIZE * (NUM_HIDDEN_LAYERS-1);
            X_end = HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS;
            Z_start = HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS;
            Z_end = HIDDEN_LAYER_SIZE * NUM_HIDDEN_LAYERS + OUTPUT_DIM;
            W_start = INPUT_DIM * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE*HIDDEN_LAYER_SIZE*(NUM_HIDDEN_LAYERS-1);
            W_end = INPUT_DIM * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE*HIDDEN_LAYER_SIZE*(NUM_HIDDEN_LAYERS-1) + HIDDEN_LAYER_SIZE*OUTPUT_DIM;
            // LOG_F(0, "%d, %d, %d, %d, %d, %d", X_start, X_end, Z_start, Z_end, W_start, W_end);
            mat_mul(params->activations, X_start, X_end, params->weights, W_start, W_end, params->biases, params->activations, Z_start, Z_end, params->logits, 1);
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
    

    for(int image = 0; image<train.size(); image++){
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
double cross_entropy_loss(std::vector<double*> pred, std::vector<double*> true_pred){
    double loss = 0.0;

    for(int i = 0; i < true_pred.size(); i++){
        double pred_clipped = std::max(*pred[i], 1e-15);
        loss += -1 * (*true_pred[i] * log(pred_clipped));  
    }
    return loss;
}

double mlp::sigmoid_der(double x){
    return x * (1.0 - x);
}


double mlp::relu_der(double x) {
    return x > 0 ? 1 : 0;
}




// // mat A is the activation of the current layer, mat D is the delta of the next layer
// // gradients = 




// return gradients;
// }


// std::vector<double*> mlp::backwards_mat_mul(std::vector<double*> mat_A, std::vector<double*> mat_D, std::vector<double> ){}

void mlp::backward_pass(network_params* params, std::vector<Data*> batch){

    
    
    std::vector<double*> features;
    features.resize(INPUT_DIM);

    int i = 0;
    int j = 0;
    int layer = 0;

    int activation_offset = params->num_activations - OUTPUT_DIM;
    int weight_offset;

    int y_size = params->true_pred.size();

    for(int image = 0; image<batch.size(); image++){
        
        convertVector(batch[image]->get_features(), features);


        // calculate error term for output neurons
        for(i = 0; i<OUTPUT_DIM; i++){
            *params->error_term[activation_offset + i] = *params->pred[y_size - OUTPUT_DIM + i] - *params->true_pred[y_size - OUTPUT_DIM + i];
        }

        //calculate the error term for last hidden layer
        int previous_error_offset = activation_offset;
        activation_offset = params->num_activations-(OUTPUT_DIM+HIDDEN_LAYER_SIZE);
        weight_offset = params->num_weights-(output_weights_size);
        for(i = 0; i<HIDDEN_LAYER_SIZE; i++){
            for(j = 0; j<OUTPUT_DIM; j++){
                *params->error_term[activation_offset + i] +=  *params->weights[weight_offset + HIDDEN_LAYER_SIZE*j + i] * *params->error_term[previous_error_offset + j];
                *params->error_term[activation_offset + i] = *params->error_term[activation_offset + i] * relu_der(*params->activations[activation_offset + i]);
            }
        }

        //calculate the error term for the other hidden layers
        for(layer = 0; layer < NUM_HIDDEN_LAYERS-1; layer++){
            previous_error_offset = activation_offset;
            activation_offset -= HIDDEN_LAYER_SIZE;
            weight_offset -= hidden_weights_size;

            // LOG_F(0, "previous layer offset : %d, activation offset: %d, weight offset : %d", previous_error_offset, activation_offset, weight_offset);
            
            for(i = 0; i<HIDDEN_LAYER_SIZE; i++){
                for(j = 0; j<HIDDEN_LAYER_SIZE; j++){
                    *params->error_term[activation_offset + i] +=  *params->weights[weight_offset + HIDDEN_LAYER_SIZE*j + i] * *params->error_term[previous_error_offset + j];
                    *params->error_term[activation_offset + i] = *params->error_term[activation_offset + i] * relu_der(*params->activations[activation_offset + i]);
                }
            }
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


int main(){

    data_handler *dh = new data_handler();
    dh->load_feature_vectors("/home/snow/learn/dl_cpp/cpp_mlp/dataset/train-images-idx3-ubyte");
    dh->load_feature_labels("/home/snow/learn/dl_cpp/cpp_mlp/dataset/train-labels-idx1-ubyte");
    dh->split_data();

    int num_classes = dh->class_counter();

    mlp *nn = new mlp();
    nn->init_network(nn->params, dh->get_train());
    
    LOG_F(0, "Train size = %ld", dh->get_train().size());
    int num_batches = dh->get_train().size()/BATCH_SIZE;

    for(int batch_num = 0; batch_num < num_batches; batch_num++){
        std::vector<Data*> batch = nn->create_batch(dh->get_train());

        nn->create_one_hot(nn->params, batch, num_classes);
        nn->forward_pass(nn->params, batch);
        LOG_F(0, "Forward pass completed for batch %d.", batch_num);
        nn->backward_pass(nn->params, batch);
        
        LOG_F(0, "Backward pass completed for batch %d.", batch_num);
        double CE_loss = cross_entropy_loss(nn->params->pred, nn->params->true_pred);
        LOG_F(0, "Loss : %f", CE_loss);
  
    }


}
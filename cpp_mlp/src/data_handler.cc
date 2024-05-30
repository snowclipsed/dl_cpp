#include "data_handler.hpp"
#include "data.hpp"
#include "loguru.hpp"

data_handler::data_handler()
{
    data = new std::vector <Data *>;
    train = new std::vector <Data *>;
    test = new std::vector <Data *>;
    validation = new std::vector <Data *>;
}

data_handler::~data_handler(){
    delete data;
    delete train;
    delete test;
    delete validation;
}

void data_handler::load_feature_vectors(std::string PATH){
    uint32_t header[4]; // stores the 4 32-bit integers before the pixel data which stores the
    unsigned char bytes[4]; // stores the 4 bytes of the 32-bit integer

    /**
     * The 4 elements in the header are:
     * 1. Magic Number
     * 2. Number of Images
     * 3. Number of Rows
     * 4. Number of Columns
     * 
    */
    FILE *file = fopen(PATH.c_str(), "r");
    if (file){
        for(int i=0; i<4; i++){
            if(fread(bytes, sizeof(bytes), 1, file)){ // if we can read 1 element of size of bytes from file stream
                header[i] = get_little_endian(bytes); //convert bytes to little endian
            }
        }
        LOG_F(0, "File header obtained: %d | %d | %d | %d", header[0], header[1], header[2], header[3]);
        /**
         * 
         * First we will read the pixel data from a file stream 
         * Then we will store the pixel data in a data object
         * Then we will store the data object into the data vector (collection of all the data objects)
         * Hence we will be able to read all the data from the file
        */
        int img_size = header[2] * header[3];
        for(unsigned int i=0; i<header[1]; i++){
            Data* d = new Data(); // initialize a data object container
            uint8_t pixel[1]; // initialize a variable to read a single pixel
            for(int j=0; j<img_size; j++){ // iterate over pixel data after the header
                if(fread(pixel, sizeof(pixel), 1, file)){ // read pixel data serially from the file stream into pixel[0]
                    d->append_features(pixel[0]); // append the pixel to the feature vector
                }
                else{
                    LOG_F(ERROR, "Error reading pixel data from file");
                    exit(1);
                }
            }
            data->push_back(d); // we then push the data object into the data vector
            // LOG_F(0, "Read pixels for image %d ", i+1);
        }
        
        if(header[1] != data->size()){
            LOG_F(ERROR, "Data size mismatch.");
            LOG_F(0, "Data size = %lu", data->size());
            LOG_F(0, "Data size should be: %d", header[1]);
        }
        LOG_F(0, "Read all data. Data size = %lu \n", data->size());
    }  
    else{
        LOG_F(ERROR, "Error opening first file");
        exit(1);
    }
}


void data_handler::load_feature_labels(std::string PATH){
    uint32_t header[2];
    unsigned char bytes[4];
    FILE *file = fopen(PATH.c_str(), "r");
    if (file){
        for(int i=0; i<2; i++){
            if(fread(bytes, sizeof(bytes), 1, file)){
                header[i] = get_little_endian(bytes);
            }else{
                LOG_F(ERROR, "Error reading label file");
                exit(1);
            }
        }
        LOG_F(0, "Label File header obtained: %d | %d", header[0], header[1]);
            for(unsigned int i=0; i<header[1]; i++){
                uint8_t label[1];
                if(fread(label, sizeof(label), 1, file)){
                        data->at(i)->set_classlabel(label[0]);
                }else{
                    LOG_F(ERROR, "Error reading label data from file");
                    exit(1);
                }
            }

            if(header[1] != data->size()){
                    LOG_F(ERROR, "Data size mismatch.");
                    LOG_F(0, "Data size = %lu", data->size());
                    LOG_F(0, "Data size should be: %d", header[1]);
                }
            LOG_F(0, "Read and stored %lu labels \n", data->size());

        }else{
        LOG_F(ERROR, "Error opening file");
        exit(1);
        }
}
void data_handler::split_data(){
    std::unordered_set<int> used_index;
    int train_size = data->size() * TRAINSET_PERCENT;
    int test_size = data->size() * TESTSET_PERCENT;
    int validation_size = data->size() * VALIDATIONSET_PERCENT;

    int count = 0;
    while(count<train_size){

        int rand_index = rand() % data->size();
        if(used_index.find(rand_index) == used_index.end()) // check if the find function returns the end of the set                                                        
                                                            //if yes then the index is not in the set
        {
            train->push_back(data->at(rand_index));
            used_index.insert(rand_index);
            count++;
        }else{continue;}
        }

    LOG_F(0, "Successfully split the data into train set");
    LOG_F(0, "Train size: %lu ", train->size());

    count = 0;
    while(count<test_size){

        int rand_index = rand() % data->size();
        if(used_index.find(rand_index) == used_index.end()) // check if the find function returns the end of the set                                                        
                                                            //if yes then the index is not in the set
        {
            test->push_back(data->at(rand_index));
            used_index.insert(rand_index);
            count++;
        }else{continue;}
    }

    LOG_F(0, "Successfully split the data into test set");
    LOG_F(0, "Test size: %lu ", test->size());

    count = 0;
    while(count<validation_size){

        int rand_index = rand() % data->size();
        if(used_index.find(rand_index) == used_index.end()) // check if the find function returns the end of the set                                                        
                                                            //if yes then the index is not in the set
        {
            validation->push_back(data->at(rand_index));
            used_index.insert(rand_index);
            count++;
        }else{continue;}
    }
    LOG_F(0, "Successfully split the data into validation set");
    LOG_F(0, "Validation size: %lu \n", validation->size());
        
}
int data_handler::class_counter(){
    int count = 0;

    for(unsigned i=0; i<data->size(); i++){
        if(class_label_map.find(data->at(i)->get_class_label()) == class_label_map.end()){
            class_label_map[data->at(i)->get_class_label()] = count;
            data->at(i)->set_enumerated_classlabel(count);
            count++;
        }
    }
    num_classes = count;
    LOG_F(0, "Class label map created");
    // LOG_F(0, "Here are the classes:");
    // for (const auto& pair : class_label_map) {
    //     LOG_F(0, "%d", pair.first);
    // }
    LOG_F(0, "Number of classes: %d \n", num_classes);
    return count;
}

uint32_t data_handler::get_little_endian(const unsigned char * bytes){
    return (uint32_t) ((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]));
}

std::vector <Data *> data_handler::get_train(){
    return *train;
}
std::vector <Data *> data_handler::get_test(){
    return *test;
}
std::vector <Data *> data_handler::get_validation(){
    return *validation;
}

// int main(){
//     data_handler *dh = new data_handler();
//     dh->load_feature_vectors("/home/snow/learn/dl_cpp/data/train-images.idx3-ubyte");
//     dh->load_feature_labels("/home/snow/learn/dl_cpp/data/train-labels.idx1-ubyte");
//     dh->split_data();
//     dh->class_counter();
// }

// g++ -std=c++11 -I./include/ -o main ./src/* to compile main!!
// ./main to run the program !!
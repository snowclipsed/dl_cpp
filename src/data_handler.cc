#include "data_handler.hpp"
#include "data.hpp"

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
        printf("File header obtained: %d %d %d %d\n", header[0], header[1], header[2], header[3]);
        /**
         * 
         * First we will read the pixel data from a file stream 
         * Then we will store the pixel data in a data object
         * Then we will store the data object into the data vector (collection of all the data objects)
         * Hence we will be able to read all the data from the file
        */
        int img_size = header[2] * header[3];
        for(int i=0; i<header[1]; i++){
            Data* d = new Data(); // initialize a data object container
            uint8_t pixel[1]; // initialize a variable to read a single pixel
            for(int j=0; j<img_size; j++){ // iterate over pixel data after the header
                if(fread(pixel, sizeof(pixel), 1, file)){ // read pixel data serially from the file stream into pixel[0]
                    d->append_features(pixel[0]); // append the pixel to the feature vector
                    printf("Read pixel as %d ", pixel[0]);
                }
                else{
                    printf("Error reading pixel data from file\n");
                    exit(1);
                }
            }
            data->push_back(d); // we then push the data object into the data vector
        }
        printf("Read all data and stored in %lu", data->size());
    }  
    else{
        printf("Error opening file\n");
        exit(1);
    }
}


void data_handler::load_feature_labels(std::string PATH){
    uint32_t header[2];
    unsigned char bytes[2];
    FILE *file = fopen(PATH.c_str(), "r");
    if (file){
        for(int i=0; i<2; i++){
            if(fread(bytes, sizeof(bytes), 1, file)){
                header[i] = get_little_endian(bytes);
            }else{
                printf("Error reading label file\n");
                exit(1);
            }
        }
        printf("Label File header obtained: %d %d\n", header[0], header[1]);
            for(int i=0; i<header[1]; i++){
                uint8_t label[1];
                if(fread(label, sizeof(label), 1, file)){
                        data->at(i)->set_classlabel(label[0]);
                        printf("Read label as %d ", label[0]);
                }else{
                    printf("Error reading label data from file\n");
                    exit(1);
                }
            }
            printf("Read all labels and stored in %lu", data->size());
        }else{
        printf("Error opening file\n");
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

    int rand_index = rand() * data->size();
    if(used_index.find(rand_index) == used_index.end()) // check if the find function returns the end of the set                                                        
                                                        //if yes then the index is not in the set
    {
        train->push_back(data->at(rand_index));
        used_index.insert(rand_index);
        count++;
    }else{continue;}
    }

printf("Successfully split the data into train set\n");
printf("Train size: %lu", train->size());

count = 0;
while(count<test_size){

    int rand_index = rand() * data->size();
    if(used_index.find(rand_index) == used_index.end()) // check if the find function returns the end of the set                                                        
                                                        //if yes then the index is not in the set
    {
        test->push_back(data->at(rand_index));
        used_index.insert(rand_index);
        count++;
    }else{continue;}
}

printf("Successfully split the data into test set\n");
printf("Test size: %lu", test->size());

count = 0;
while(count<validation_size){

    int rand_index = rand() * data->size();
    if(used_index.find(rand_index) == used_index.end()) // check if the find function returns the end of the set                                                        
                                                        //if yes then the index is not in the set
    {
        validation->push_back(data->at(rand_index));
        used_index.insert(rand_index);
        count++;
    }else{continue;}
}
printf("Successfully split the data into validation set\n");
printf("Validation size: %lu", validation->size());
        
}
void data_handler::class_counter(){
    int count = 0;

    for(unsigned i=0; i<data->size(); i++){
        if(class_label_map.find(data->at(i)->get_class_label()) == class_label_map.end()){
            class_label_map[data->at(i)->get_class_label()] = count;
            data->at(i)->set_enumerated_classlabel(count);
            count++;
    }
}

uint32_t get_little_endian(const unsigned char * bytes);

std::vector <Data *> get_train();
std::vector <Data *> get_test();
std::vector <Data *> get_validation();

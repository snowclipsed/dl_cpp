#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include <fstream>
#include "stdint.h"
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>



class data_handler {
    
    /*
    This class will load and split the data into train test and validation.
    */
    std::vector <Data *> data; // all the data stored in this variable
    
    // we will then split into the corresponding train test and validation sets

    std::vector <Data *> train;
    std::vector <Data *> test;
    std::vector <Data *> validation;


    int num_classes;
    int feature_vector_size;
    std::map <uint8_t, int> class_label_map; // enumerated labels will be mapped to the uint8_t labels

    const double TRAINSET_PERCENT = 0.75;
    const double TESTSET_PERCENT = 0.10;
    const double VALIDATIONSET_PERCENT = 0.05;

    public:
    data_handler();
    ~data_handler();

    void load_feature_vectors(std::string PATH);
    void load_feature_labels(std::string PATH);
    void split_data();
    void class_counter();

    uint32_t get_little_endian(const unsigned char * bytes);

    std::vector <Data *> get_train();
    std::vector <Data *> get_test();
    std::vector <Data *> get_validation();


};


#endif
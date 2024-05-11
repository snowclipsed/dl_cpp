#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include "stdint.h"
#include "stdio.h"


class Data {

    /*
    This is a storage class for the MNIST data.
    R = Row Size = 28
    C = Column Size = 28
    L = Size of Label = Unsigned Byte (uint8_t)
    E = Size of enumerated label = integer value from labels
    
    */

    std::vector <uint8_t> * features_RC; // It is a pointer to the feature vector. Size of vector = Row Size * Column Size.
    uint8_t class_label_L;
    int enum_label_E;


    public:
    Data();
    ~Data();
    void set_features(std::vector <uint8_t> *);
    void set_classlabel(uint8_t);
    void set_enumerated_classlabel(int);
    void append_features(uint8_t);

    int get_feature_vector_size();
    uint8_t get_class_label();
    int get_enumerated_class_label();


    // getters
    std::vector<uint8_t> * get_features();

};

#endif
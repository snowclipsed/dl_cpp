#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include "stdint.h"
#include "stdio.h"


class Data {

    /*
    This is a storage class for the MNIST data.
    
    */

    std::vector <uint8_t> * features; // this does not include the class
    uint8_t classlabel;
    int enumerated_classlabel;


    public:
    void set_features(std::vector <uint8_t> *);
    void set_classlabel(uint8_t);
    void set_enumerated_classlabel(int);
    void append_features(uint8_t);


    // getters
    std::vector<uint8_t> * get_features();

};

#endif
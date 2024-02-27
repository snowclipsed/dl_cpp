#include "data.hpp"
#include "data_handler.hpp"

#include <fstream>

void Data::set_features(std::vector <uint8_t> *vector){
    features = vector;
}
void Data::set_classlabel(uint8_t label){
    classlabel = label;
}
void Data::set_enumerated_classlabel(int label){
    enumerated_classlabel = label;
}

void Data::append_features(uint8_t feature){
    features->push_back(feature);
}

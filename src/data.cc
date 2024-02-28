#include "data.hpp"
#include "data_handler.hpp"

#include <fstream>

void Data::set_features(std::vector <uint8_t> *vector){
    features_RC = vector; // setter for feature vector
}


void Data::set_classlabel(uint8_t label){
    class_label_L = label; // setter for class label
}
void Data::set_enumerated_classlabel(int label){
    enum_label_E = label; // setter for enumerated class label
}

void Data::append_features(uint8_t feature){
    features_RC->push_back(feature); // append feature to the feature vector
}


int Data::get_feature_vector_size(){
    return features_RC->size(); // return the size of the feature vector
}

uint8_t Data::get_class_label(){
    return class_label_L; // return the class label
}

uint8_t Data::get_enumerated_class_label(){
    return enum_label_E; // return the enumerated class label
}

std::vector<uint8_t> * Data::get_features(){

    return features_RC; // return the feature vector

}


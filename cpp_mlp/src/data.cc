#include "data.hpp"
#include "data_handler.hpp"
#include "loguru.hpp"

#include <fstream>

Data::Data(){
    features_RC = new std::vector <uint8_t>; // initialize the feature vector
    features_RC_double = new std::vector <double>;
}

Data::~Data(){
    delete features_RC; // delete the feature vector
    // delete features_RC_double;
}

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

int Data::get_enumerated_class_label(){
    return enum_label_E; // return the enumerated class label
    LOG_F(0, "LABEL : %d", enum_label_E);
}

std::vector<uint8_t> * Data::get_features(){

    return features_RC; // return the feature vector

}

std::vector<double> * Data::get_features_double(){
    return features_RC_double;
}

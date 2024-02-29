#datasets
The dataset for MNSIT is stored in a binary file format. Original files by YLeCun are stored in high endian format. 
# Train Labels


| DType         | Description                       |
| ------------- | --------------------------------- |
| 32 bit Int    | Magic Number for Byte Checking    |
| 32 bit Int    | Number of Items                   |
| unsigned byte | Image Label                       |
|               | Image label Continues for N items |
# Train Images

| DType         | Description                        |
| ------------- | ---------------------------------- |
| 32 bit Int    | Magic Number for Byte Checking     |
| 32 bit Int    | Number of Items                    |
| 32 bit int    | row size(28)                       |
| 32 bit int    | column size (28)                   |
| unsigned byte | Image Pixels                       |
|               | Image label Continues for N pixels |

# Data Class

We store our dataset in this class. This class is used to load our data and make it accessible and workable by the algorithms we apply on it.

```hpp
class Data {

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
```

`features` stores the feature vector for each class
`classlabel` stores the image's class label
`enumerated_classlabel` stores the numerical value to which the classlabel corresponds to


# Data Loader

```hpp
#include <fstream>
#include "stdint.h"
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
```

- We need `fstream` to read from the data file.
- We need `stdint` for the integer datatypes
- We need `data.hpp` for the data storage class we defined
- `string` is needed for the enumerated labels
- `map` is needed to map the enumerated string labels to the actual int labels
- `unordered_set` is used to index the files while splitting into train and test

Steps
1. Load the data as vector of data class
2. 
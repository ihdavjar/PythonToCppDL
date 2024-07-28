# Description
1) Each folder contains files required to run the respective model.
2) In order to compile and run the model use the following commands

```shell
g++ *.cpp -o model_name
./model_name
```
3) Make sure you have installed the g++ compiler.
4) A report.pdf is also included with detailed description.

## Contents of each folder
Each folder contains:
```
layers.cpp, layers.h
model.cpp, model.h
test_bench.cpp, test_data/test_image.h
weights.h
```

### layers.cpp, layers.h
These files contains the functions/layers declarations which can be used to build any neural network.

### model.cpp, model.h
These files contains the model implementation utilising the functions/layers implemented in the layers.cpp

### test_bench.cpp
Contains code to run model on image present in `test_image.h`.

### weights.h
This contains all the weights involved in the respective model.


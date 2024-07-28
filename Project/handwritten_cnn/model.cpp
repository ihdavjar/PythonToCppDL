#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include <string.h>
#include <climits>
#include <iostream>
#include "model.h"
#include "layers.h"
#include "weights.h"

// inputs = tf.keras.Input(shape=input_shape)
// x = tf.keras.layers.Conv2D(filters=8, kernel_size=5, activation='relu')(inputs)
// x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
// x = tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu')(x)
// x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
// x = tf.keras.layers.Flatten()(x)
// x = tf.keras.layers.Dense(units=128, activation='relu')(x)
// x = tf.keras.layers.Dropout(0.2)(x)
// outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)

// Input Size = 28x28x1
// Output Size = 10

void handwritten_cnn(double flattenedImage[], int imageWidth, int imageHeight, int imageDepth, double output[]){

    // Calculate the output of the first convolutional layer
    double conv1_output[24*24*8];
    convolution(1, 8, 5, 5, 1, 1, 0, 0, 28, 28, flattenedImage, quant_conv2d_weights, quant_conv2d_bias, conv1_output);
    relu(conv1_output, conv1_output, 24*24*8);

    // Max pooling layer 1 -> pool size = 2x2, stride = 2
    double maxpool1_output[12*12*8];
    maxPooling(conv1_output, maxpool1_output, 24, 24, 8, 2, 2, 2, 2, 0, 0);

    double conv2_output[8*8*16];
    convolution(8, 16, 5, 5, 1, 1, 0, 0, 12, 12, maxpool1_output, quant_conv2d_1_weights, quant_conv2d_1_bias, conv2_output);
    relu(conv2_output, conv2_output, 8*8*16);

    double maxpool2_output[4*4*16];
    maxPooling(conv2_output, maxpool2_output, 8, 8, 16, 2, 2, 2, 2, 0, 0);

    double dense1_output[128];
    linear(maxpool2_output, dense1_output, quant_dense_weights, quant_dense_bias, 4*4*16, 128);
    relu(dense1_output, dense1_output, 128);

    double dense2_output[10];
    linear(dense1_output, dense2_output, quant_dense_1_weights, quant_dense_1_bias, 128, 10);
    softmax(dense2_output, output, 10);
}
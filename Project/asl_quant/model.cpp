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

void asl_quant(double flattenedImage[], int imageWidth, int imageHeight, int imageDepth, double output[]){

    // Calculate the output of the first convolutional layer
    // Convolutional layer 1 -> 32 , 3x3 filters, activation = relu, stride = 1, padding = same
    // Input shape = 64x64x3
    double *conv1_out = new double[64*64*32];
    convolution(3, 32, 3, 3, 1, 1, 0, 0, 64, 64, flattenedImage, quant_conv2d_weights, quant_conv2d_bias, conv1_out);
    relu(conv1_out, conv1_out, 62*62*32);
    
    //Convolutional layer 2 -> 64 , 3x3 filters, activation = relu, stride = 1, padding = same 
    double *conv2_out = new double[64*64*64];
    convolution(32, 64, 3, 3, 1, 1, 0, 0, 64, 64, conv1_out, quant_conv2d_1_weights, quant_conv2d_1_bias, conv2_out);
    relu(conv2_out, conv2_out, 62*62*64);

    // Max pooling layer 1 -> pool size = 2x2, stride = 2
    double *maxpool1_out = new double[32*32*64];
    maxPooling(conv2_out, maxpool1_out, 64, 64, 64, 2, 2, 2, 2, 0, 0);

    // Convolutional layer 3 -> 64 , 3x3 filters, activation = relu, stride = 1, padding = same
    double *conv3_out = new double[32*32*64];
    convolution(64, 64, 3, 3, 1, 1, 0, 0, 32, 32, maxpool1_out, quant_conv2d_2_weights, quant_conv2d_2_bias, conv3_out);
    relu(conv3_out, conv3_out, 30*30*64);

    // Max pooling layer 2 -> pool size = 2x2, stride = 2
    double *maxpool2_out = new double[16*16*64];
    maxPooling(conv3_out, maxpool2_out, 32, 32, 64, 2, 2, 2, 2, 0, 0);

    // Convolutional layer 4 -> 128 , 3x3 filters, activation = relu, stride = 1, padding = same
    double *conv4_out = new double[16*16*128];
    convolution(64, 128, 3, 3, 1, 1, 0, 0, 16, 16, maxpool2_out, quant_conv2d_3_weights, quant_conv2d_3_bias, conv4_out);

    // Max pooling layer 3 -> pool size = 2x2, stride = 2
    double *maxpool3_out = new double[8*8*128];
    maxPooling(conv4_out, maxpool3_out, 16, 16, 128, 2, 2, 2, 2, 0, 0);

    // Dense layer 1 -> 526 units, activation = relu
    double *dense1_out = new double[526];
    linear(maxpool3_out, dense1_out, quant_dense_weights, quant_dense_bias, 8*8*128, 526);
    relu(dense1_out, dense1_out, 526);

    // Dense layer 2 -> 128 units, activation = relu
    double *dense2_out = new double[128];
    linear(dense1_out, dense2_out, quant_dense_1_weights, quant_dense_1_bias, 526, 128);
    relu(dense2_out, dense2_out, 128);

    // Dense layer 3 -> 29 units, activation = relu
    double *dense3_out = new double[29];
    linear(dense2_out, dense3_out, quant_dense_2_weights, quant_dense_2_bias, 128, 29);
    softmax(dense3_out, output, 29);

}
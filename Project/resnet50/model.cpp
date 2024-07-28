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



void block_2(int c_in, int c_out, double input[], double output[], const double conv1_conv_weights[], const double conv1_conv_bias[], const double conv2_conv_weights[], const double conv2_conv_bias[], const double conv3_conv_weights[], const double conv3_conv_bias[]){
    
    // Allocate memory for the output of the first convolutional layer using malloc
    double* conv1Output = (double*)malloc(56*56*c_out*sizeof(double));

    convolution(c_in, c_out, 1, 1, 1, 1, 0, 0, 56, 56, input, conv1_conv_weights, conv1_conv_bias, conv1Output);
    relu(conv1Output, conv1Output, 56*56*c_out);

    double* conv2Output = (double*)malloc(56*56*c_out*sizeof(double));
    convolution(c_out, c_out, 3, 3, 1, 1, 1, 1, 56, 56, conv1Output, conv2_conv_weights, conv2_conv_bias, conv2Output);
    free(conv1Output);
    relu(conv2Output, conv2Output, 56*56*c_out);

    double* conv3Output = (double*)malloc(56*56*c_out*4*sizeof(double));
    convolution(c_out, c_out*4, 1, 1, 1, 1, 0, 0, 56, 56, conv2Output, conv3_conv_weights, conv3_conv_bias, conv3Output);
    free(conv2Output);
    relu(conv3Output, output, 56*56*c_out*4);

}


void block_3(int c_in, int c_out, int stride, double input[], double output[], const double conv1_conv_weights[], const double conv1_conv_bias[], const double conv2_conv_weights[], const double conv2_conv_bias[], const double conv3_conv_weights[], const double conv3_conv_bias[]){
    
    double* conv1Output = (double*)malloc(28*28*c_out*sizeof(double));
    convolution(c_in, c_out, 1, 1, 1, 1, 0, 0, 28, 28, input, conv1_conv_weights, conv1_conv_bias, conv1Output);
    relu(conv1Output, conv1Output, 28*28*c_out);

    double* conv2Output = (double*)malloc(28*28*c_out*sizeof(double));
    convolution(c_out, c_out, 3, 3, stride, stride, 1, 1, 28, 28, conv1Output, conv2_conv_weights, conv2_conv_bias, conv2Output);
    free(conv1Output);
    relu(conv2Output, conv2Output, 28*28*c_out);

    double* conv3Output = (double*)malloc(28*28*c_out*4*sizeof(double));
    convolution(c_out, c_out*4, 1, 1, 1, 1, 0, 0, 28, 28, conv2Output, conv3_conv_weights, conv3_conv_bias, conv3Output);
    free(conv2Output);
    relu(conv3Output, output, 28*28*c_out*4);

}

void block_4(int c_in, int c_out, int stride, double input[], double output[], const double conv1_conv_weights[], const double conv1_conv_bias[], const double conv2_conv_weights[], const double conv2_conv_bias[], const double conv3_conv_weights[], const double conv3_conv_bias[]){
    
    double* conv1Output = (double*)malloc(14*14*c_out*sizeof(double));
    convolution(c_in, c_out, 1, 1, 1, 1, 0, 0, 14, 14, input, conv1_conv_weights, conv1_conv_bias, conv1Output);
    relu(conv1Output, conv1Output, 14*14*c_out);


    double* conv2Output = (double*)malloc(14*14*c_out*sizeof(double));
    convolution(c_out, c_out, 3, 3, stride, stride, 1, 1, 14, 14, conv1Output, conv2_conv_weights, conv2_conv_bias, conv2Output);
    free(conv1Output);
    relu(conv2Output, conv2Output, 14*14*c_out);


    double* conv3Output = (double*)malloc(14*14*c_out*4*sizeof(double));
    convolution(c_out, c_out*4, 1, 1, 1, 1, 0, 0, 14, 14, conv2Output, conv3_conv_weights, conv3_conv_bias, conv3Output);
    free(conv2Output);
    relu(conv3Output, output, 14*14*c_out*4);


}

void block_5(int c_in, int c_out, int stride, double input[], double output[], const double conv1_conv_weights[], const double conv1_conv_bias[], const double conv2_conv_weights[], const double conv2_conv_bias[], const double conv3_conv_weights[], const double conv3_conv_bias[]){
    
    double* conv1Output = (double*)malloc(7*7*c_out*sizeof(double));
    convolution(c_in, c_out, 1, 1, 1, 1, 0, 0, 7, 7, input, conv1_conv_weights, conv1_conv_bias, conv1Output);
    relu(conv1Output, conv1Output, 7*7*c_out);

    double* conv2Output = (double*)malloc(7*7*c_out*sizeof(double));
    convolution(c_out, c_out, 3, 3, stride, stride, 1, 1, 7, 7, conv1Output, conv2_conv_weights, conv2_conv_bias, conv2Output);
    free(conv1Output);
    relu(conv2Output, conv2Output, 7*7*c_out);

    double* conv3Output = (double*)malloc(7*7*c_out*4*sizeof(double));
    convolution(c_out, c_out*4, 1, 1, 1, 1, 0, 0, 7, 7, conv2Output, conv3_conv_weights, conv3_conv_bias, conv3Output);
    free(conv2Output);
    relu(conv3Output, output, 7*7*c_out*4);

}


void ResNet50(double flattenedImage[], int imageWidth, int imageHeight, int imageDepth, double output[]) {

    // Adding something to check if the code is running
    
    // ############################## Layer 1 ##############################
    double* conv1Output = new double[112*112*64];
    convolution(imageDepth, 64, 7, 7, 2, 2, 3, 3, imageWidth, imageHeight, flattenedImage, conv1_conv_weights, conv1_conv_bias, conv1Output);
    relu(conv1Output, conv1Output, 112*112*64);
    double* maxpool1Output = new double[56*56*64];
    maxPooling(conv1Output, maxpool1Output, 64, 112, 112, 3, 3, 2, 2, 1, 1);

    
    // ############################## Layer 2 ##############################
    // block 1
    double* block2_1_Output = new double[56*56*256];
    block_2(64, 64, maxpool1Output, block2_1_Output, conv2_block1_1_conv_weights, conv2_block1_1_conv_bias, conv2_block1_2_conv_weights, conv2_block1_2_conv_bias, conv2_block1_3_conv_weights, conv2_block1_3_conv_bias);

    // downsample
    double* downsample0Output = new double[56*56*256];
    convolution(64, 256, 1, 1, 1, 1, 0, 0, 56, 56, maxpool1Output, conv2_block1_0_conv_weights, conv2_block1_0_conv_bias, downsample0Output);
    relu(downsample0Output, downsample0Output, 56*56*256);
    // Residual Connection
    add(downsample0Output, block2_1_Output, block2_1_Output, 56*56*256);
    free(downsample0Output);
    
    // Block 2
    double* block2_2_Output = new double[56*56*256];
    block_2(256, 64, block2_1_Output, block2_2_Output, conv2_block2_1_conv_weights, conv2_block2_1_conv_bias, conv2_block2_2_conv_weights, conv2_block2_2_conv_bias, conv2_block2_3_conv_weights, conv2_block2_3_conv_bias);
    // Residual Connection
    add(block2_1_Output, block2_2_Output, block2_2_Output, 56*56*256);
    free(block2_2_Output);


    // // Block 3
    double* block2_3_Output = new double[56*56*256];
    block_2(256, 64, block2_2_Output, block2_3_Output, conv2_block3_1_conv_weights, conv2_block3_1_conv_bias, conv2_block3_2_conv_weights, conv2_block3_2_conv_bias, conv2_block3_3_conv_weights, conv2_block3_3_conv_bias);
    // Residual Connection
    add(block2_2_Output, block2_3_Output, block2_3_Output, 56*56*256);
    free(block2_2_Output);


    // ############################## Layer 3 ##############################
    // block 1
    double* block3_1_Output = new double[28*28*512];
    block_3(256, 128, 2,block2_3_Output, block3_1_Output, conv3_block1_1_conv_weights, conv3_block1_1_conv_bias, conv3_block1_2_conv_weights, conv3_block1_2_conv_bias, conv3_block1_3_conv_weights, conv3_block1_3_conv_bias);
    // downsample
    double* downsample1Output = new double[28*28*512];
    convolution(256, 512, 1, 1, 1, 1, 0, 0, 28, 28, block2_3_Output, conv3_block1_0_conv_weights, conv3_block1_0_conv_bias, downsample1Output);
    relu(downsample1Output, downsample1Output, 28*28*512);
    // Residual Connection
    add(downsample1Output, block3_1_Output, block3_1_Output, 28*28*512);
    free(block2_3_Output);
    free(downsample1Output);

    // Block 2
    double* block3_2_Output = new double[28*28*512];
    block_3(512, 128, 1,block3_1_Output, block3_2_Output, conv3_block2_1_conv_weights, conv3_block2_1_conv_bias, conv3_block2_2_conv_weights, conv3_block2_2_conv_bias, conv3_block2_3_conv_weights, conv3_block2_3_conv_bias);
    // Residual Connection
    add(block3_1_Output, block3_2_Output, block3_2_Output, 28*28*512);
    free(block3_1_Output);

    // Block 3
    double* block3_3_Output = new double[28*28*512];
    block_3(512, 128, 1,block3_2_Output, block3_3_Output, conv3_block3_1_conv_weights, conv3_block3_1_conv_bias, conv3_block3_2_conv_weights, conv3_block3_2_conv_bias, conv3_block3_3_conv_weights, conv3_block3_3_conv_bias);   
    // Residual Connection
    add(block3_2_Output, block3_3_Output, block3_2_Output, 28*28*512);
    free(block3_2_Output);

    // Block 4
    double* block3_4_Output = new double[28*28*512];
    block_3(512, 128, 1,block3_3_Output, block3_4_Output, conv3_block4_1_conv_weights, conv3_block4_1_conv_bias, conv3_block4_2_conv_weights, conv3_block4_2_conv_bias, conv3_block4_3_conv_weights, conv3_block4_3_conv_bias);
    // Residual Connection
    add(block3_3_Output, block3_4_Output, block3_4_Output, 28*28*512);
    free(block3_3_Output);


    // ############################## Layer 4 ##############################
    // block 1
    double* block4_1_Output = new double[14*14*1024];
    block_4(512, 256, 2, block3_4_Output, block4_1_Output, conv4_block1_1_conv_weights, conv4_block1_1_conv_bias, conv4_block1_2_conv_weights, conv4_block1_2_conv_bias, conv4_block1_3_conv_weights, conv4_block1_3_conv_bias);
    // downsample
    double* downsample2Output = new double[14*14*1024];
    convolution(512, 1024, 1, 1, 2, 2, 0, 0, 14, 14, block3_4_Output, conv4_block1_0_conv_weights, conv4_block1_0_conv_bias, downsample2Output);
    relu(downsample2Output, downsample2Output, 14*14*1024);
    // Residual Connection
    add(downsample2Output, block4_1_Output, block4_1_Output, 14*14*1024);
    free(downsample2Output);

    // Block 2
    double* block4_2_Output = new double[14*14*1024];
    block_4(1024, 256, 1, block4_1_Output, block4_2_Output, conv4_block2_1_conv_weights, conv4_block2_1_conv_bias, conv4_block2_2_conv_weights, conv4_block2_2_conv_bias, conv4_block2_3_conv_weights, conv4_block2_3_conv_bias);
    // Residual Connection
    add(block4_1_Output, block4_2_Output, block4_2_Output, 14*14*1024);
    free(block4_1_Output);

    // Block 3
    double* block4_3_Output = new double[14*14*1024];
    block_4(1024, 256, 1, block4_2_Output, block4_3_Output, conv4_block3_1_conv_weights, conv4_block3_1_conv_bias, conv4_block3_2_conv_weights, conv4_block3_2_conv_bias, conv4_block3_3_conv_weights, conv4_block3_3_conv_bias);
    // Residual Connection
    add(block4_2_Output, block4_3_Output, block4_3_Output, 14*14*1024);
    free(block4_2_Output);

    // Block 4
    double* block4_4_Output = new double[14*14*1024];
    block_4(1024, 256, 1, block4_3_Output, block4_4_Output, conv4_block4_1_conv_weights, conv4_block4_1_conv_bias, conv4_block4_2_conv_weights, conv4_block4_2_conv_bias, conv4_block4_3_conv_weights, conv4_block4_3_conv_bias);
    // Residual Connection
    add(block4_3_Output, block4_4_Output, block4_4_Output, 14*14*1024);
    free(block4_3_Output);

    // Block 5
    double* block4_5_Output = new double[14*14*1024];
    block_4(1024, 256, 1, block4_4_Output, block4_5_Output, conv4_block5_1_conv_weights, conv4_block5_1_conv_bias, conv4_block5_2_conv_weights, conv4_block5_2_conv_bias, conv4_block5_3_conv_weights, conv4_block5_3_conv_bias);
    // Residual Connection
    add(block4_4_Output, block4_5_Output, block4_5_Output, 14*14*1024);
    free(block4_4_Output);

    // Block 6
    double* block4_6_Output = new double[14*14*1024];
    block_4(1024, 256, 1, block4_5_Output, block4_6_Output, conv4_block6_1_conv_weights, conv4_block6_1_conv_bias, conv4_block6_2_conv_weights, conv4_block6_2_conv_bias, conv4_block6_3_conv_weights, conv4_block6_3_conv_bias);
    // Residual Connection
    add(block4_5_Output, block4_6_Output, block4_6_Output, 14*14*1024);
    free(block4_5_Output);


    // ############################## Layer 5 ##############################
    // block 1
    double* block5_1_Output = new double[7*7*2048];
    block_5(1024, 512, 2, block4_6_Output, block5_1_Output, conv5_block1_1_conv_weights, conv5_block1_1_conv_bias, conv5_block1_2_conv_weights, conv5_block1_2_conv_bias, conv5_block1_3_conv_weights, conv5_block1_3_conv_bias);
    // downsample
    double* downsample3Output = new double[7*7*2048];
    convolution(1024, 2048, 1, 1, 2, 2, 0, 0, 7, 7, block4_6_Output, conv5_block1_0_conv_weights, conv5_block1_0_conv_bias, downsample3Output);
    relu(downsample3Output, downsample3Output, 7*7*2048);
    // Residual Connection
    add(downsample3Output, block5_1_Output, block5_1_Output, 7*7*2048);
    free(downsample3Output);

    // Block 2
    double* block5_2_Output = new double[7*7*2048];
    block_5(2048, 512, 1, block5_1_Output, block5_2_Output, conv5_block2_1_conv_weights, conv5_block2_1_conv_bias, conv5_block2_2_conv_weights, conv5_block2_2_conv_bias, conv5_block2_3_conv_weights, conv5_block2_3_conv_bias);
    // Residual Connection
    add(block5_1_Output, block5_2_Output, block5_2_Output, 7*7*2048);
    free(block5_1_Output);

    // Block 3
    double* block5_3_Output = new double[7*7*2048];
    block_5(2048, 512, 1, block5_2_Output, block5_3_Output, conv5_block3_1_conv_weights, conv5_block3_1_conv_bias, conv5_block3_2_conv_weights, conv5_block3_2_conv_bias, conv5_block3_3_conv_weights, conv5_block3_3_conv_bias);
    // Residual Connection
    add(block5_2_Output, block5_3_Output, block5_3_Output, 7*7*2048);
    free(block5_2_Output);

    // ############################## Layer 6 ##############################
    // Global Average Pooling
    double* avgpool1Output = new double[1*1*2048];
    avgPooling(block5_3_Output, avgpool1Output, 7, 7, 2048, 7, 7, 1, 1, 0, 0);
    free(block5_3_Output);
    

    // Prediction Layer
    double *dense1Output = new double[1000];
    linear(avgpool1Output, dense1Output, predictions_weights, predictions_bias, 2048, 1000);
    relu(dense1Output, dense1Output, 1000);

    // Dense Layer
    double* dense2Output = new double[2];
    linear(dense1Output, dense2Output, dense_weights, dense_bias, 1000, 2);
    softmax(dense2Output, output, 2);
    
}

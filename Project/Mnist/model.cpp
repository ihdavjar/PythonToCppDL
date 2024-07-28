#include "layers.h"
#include "weights.h"
#include <iostream>
using namespace std;

void CNN(double flattenedImage[], int imageWidth, int imageHeight, double output[]) {
    #pragma HLS INTERFACE s_axilite port=imageWidth bundle=control
    #pragma HLS INTERFACE s_axilite port=imageHeight bundle=control
    #pragma HLS INTERFACE s_axilite port=flattenedImage bundle=control
    #pragma HLS INTERFACE s_axilite port=output bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Convolutional layer1
    int num_kernels1 = 84;
    int kernelWidth1 = 3;
    int kernelHeight1 = 3;
    int imagedepth1 = 1;
    double conv1Output[26*26*84];
    convolution(1, 84, 3, 3, 1, 1, 0, 0, 28, 28, flattenedImage, convolution1_weights, convolution1_bias, conv1Output);
    relu(conv1Output, conv1Output, 26*26*84);
    addNoise(conv1Output, 0.078735, 26*26*84);

    // Convolutional layer2
    int num_kernels2 = 32;
    int kernelWidth2 = 3;
    int kernelHeight2 = 3;
    int imagedepth2 = 84;
    double conv2Output[24*24*32];
    convolution(84, 32, 3, 3, 1, 1, 0, 0, 26, 26, conv1Output, convolution2_weights, convolution2_bias, conv2Output);
    relu(conv2Output, conv2Output, 24*24*32);
    addNoise(conv2Output, 0.046218, 24*24*32);

    // convulation 3
    double conv3Output[22*22*132];
    convolution(32, 132, 3, 3, 1, 1, 0, 0, 24, 24, conv2Output, convolution3_weights, convolution3_bias, conv3Output);
    relu(conv3Output, conv3Output, 22*22*132);
    addNoise(conv3Output, 0.089512, 22*22*132);

    // convulation 4
    double conv4Output[20*20*64];
    convolution(132, 64, 3, 3, 1, 1, 0, 0, 22, 22, conv3Output, convolution4_weights, convolution4_bias, conv4Output);
    relu(conv4Output, conv4Output, 20*20*64);
    addNoise(conv4Output, 0.027931, 20*20*64);

    // convolution 5
    double conv5Output[18*18*118];
    convolution(64, 118, 3, 3, 1, 1, 0, 0, 20, 20, conv4Output, convolution5_weights, convolution5_bias, conv5Output);
    relu(conv5Output, conv5Output, 18*18*118);
    addNoise(conv5Output, 0.053142, 18*18*118);

    // fully connected layer.
    double fully1output[10];
    linear(conv5Output, fully1output, dense1_weights, dense1_bias, 38232, 10);

    softmax(fully1output, output, 10);
}
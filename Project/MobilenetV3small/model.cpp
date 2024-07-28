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


// Used in Block1
void block_0(int c_in, int c_out, int c_squeeze, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int image_h, int image_w, double flattenedImage[], const double conv1_weights[], const double conv1_bias[], const double conv2_weights[], const double conv2_bias[], const double conv3_weights[], const double conv3_bias[], const double conv4_weights[], const double conv4_bias[], double output[]){
    // Depthwise Convolution (Depthwise Convolution)
    // New Image Size
    int new_image_h = (image_h + 2*p_h - k_h)/s_h+1;
    int new_image_w = (image_w + 2*p_w - k_w)/s_w+1;

    double* conv1Output = (double*)malloc(new_image_h*new_image_w*c_in*sizeof(double));
    depthwise_convolution(c_in, k_h, k_w, s_h, s_w, p_h, p_w, image_h, image_w, flattenedImage, conv1_weights, conv1_bias, conv1Output);
    relu(conv1Output, conv1Output, new_image_h*new_image_w*c_in);

    // Squeeze and Excitation (Squeeze and Excitation)
    double* conv2Output = (double*)malloc(1*1*c_in*sizeof(double));
    avgPooling(conv1Output, conv2Output, new_image_h, new_image_w, c_in, new_image_h, new_image_w, 1, 1, 0, 0);

    double* conv3Output = (double*)malloc(1*1*c_squeeze*sizeof(double));
    convolution(c_in, c_squeeze, 1, 1, 1, 1, 0, 0, 1, 1, conv2Output, conv2_weights, conv2_bias, conv3Output);
    relu(conv3Output, conv3Output, 1*1*c_squeeze);

    double* conv4Output = (double*)malloc(1*1*c_in*sizeof(double));
    convolution(c_squeeze, c_in, 1, 1, 1, 1, 0, 0, 1, 1, conv3Output, conv3_weights, conv3_bias, conv4Output);
    hard_swish(conv4Output, conv4Output, 1*1*c_in);

    // 1*1*c_mid has to be multiplied with image conv1Output
    for (int i = 0; i<c_in; i++){
        for (int j = 0; j<new_image_h*new_image_w; j++){
            conv1Output[i*new_image_h*new_image_w+j] = conv1Output[i*new_image_h*new_image_w+j]*conv4Output[i];
        }
    }

    convolution(c_in, c_out, 1, 1, 1, 1, 0, 0, new_image_h, new_image_w, conv1Output, conv4_weights, conv4_bias, output);

    free(conv1Output);
    free(conv2Output);
    free(conv3Output);
    free(conv4Output);
}


void block_1(int c_in, int c_out, int c_mid, int c_squeeze, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int image_h, int image_w, double flattenedImage[], const double conv1_weights[], const double conv1_bias[], const double conv2_weights[], const double conv2_bias[], const double conv3_weights[], const double conv3_bias[], const double conv4_weights[], const double conv4_bias[], const double conv5_weights[], const double conv5_bias[], double output[]) {
    
    // First Convolution (Expand)
    double* conv1Output = (double*)malloc(image_h*image_w*c_mid*sizeof(double));
    convolution(c_in, c_mid, 1, 1, 1, 1, 0, 0, image_h, image_w, flattenedImage, conv1_weights, conv1_bias, conv1Output);
    hard_swish(conv1Output, conv1Output, image_h*image_w*c_mid);

    // Depthwise Convolution (Depthwise)
    // New Image Size
    int new_image_h = (image_h + 2*p_h - k_h)/s_h+1;
    int new_image_w = (image_w + 2*p_w - k_w)/s_w+1;

    double* conv2Output = (double*)malloc(new_image_h*new_image_w*c_mid*sizeof(double));
    depthwise_convolution(c_mid, k_h, k_w, s_h, s_w, p_h, p_w, image_h, image_w, conv1Output, conv2_weights, conv2_bias, conv2Output);
    hard_swish(conv2Output, conv2Output, new_image_h*new_image_w*c_mid);

    // Squeeze and Excitation (Squeeze and Excitation)
    double* conv3Output = (double*)malloc(1*1*c_mid*sizeof(double));
    avgPooling(conv2Output, conv3Output, new_image_h, new_image_w, c_mid, new_image_h, new_image_w, 1, 1, 0, 0);
    
    double* conv4Output = (double*)malloc(1*1*c_squeeze*sizeof(double));
    convolution(c_mid, c_squeeze, 1, 1, 1, 1, 0, 0, 1, 1, conv3Output, conv3_weights, conv3_bias, conv4Output);
    relu(conv4Output, conv4Output, 1*1*c_squeeze);

    double* conv5Output = (double*)malloc(1*1*c_mid*sizeof(double));
    convolution(c_squeeze, c_mid, 1, 1, 1, 1, 0, 0, 1, 1, conv4Output, conv4_weights, conv4_bias, conv5Output);
    hard_swish(conv5Output, conv5Output, 1*1*c_mid);

    // 1*1*c_mid has to be multiplied with image conv1Output
    for (int i = 0; i<c_mid; i++){
        for (int j = 0; j<new_image_h*new_image_w; j++){
            conv2Output[i*new_image_h*new_image_w+j] = conv2Output[i*new_image_h*new_image_w+j]*conv5Output[i];
        }
    }

    // // Output Convolutions (Project)
    convolution(c_mid, c_out, 1, 1, 1, 1, 0, 0, new_image_h, new_image_w, conv2Output, conv5_weights, conv5_bias, output);

    free(conv1Output);
    free(conv2Output);
    free(conv3Output);
    free(conv4Output);
    free(conv5Output);
}

// Used in Block2, Block3
// Without Squueze and Excitation
void block_2(int c_in, int c_out, int c_mid, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int image_h, int image_w, double flattenedImage[], double output[], const double conv1_weights[], const double conv1_bias[], const double conv2_weights[], const double conv2_bias[], const double conv3_weights[], const double conv3_bias[]) {
    
    // First Convolution (Expand)
    double* conv1Output = (double*)malloc(image_h*image_w*c_mid*sizeof(double));
    convolution(c_in, c_mid, 1, 1, 1, 1, 0, 0, image_h, image_w, flattenedImage, conv1_weights, conv1_bias, conv1Output);
    relu(conv1Output, conv1Output, image_h*image_w*c_mid);

    // Depthwise Convolution (Depthwise)
    // New image size
    int new_image_h = (image_h + 2*p_h - k_h)/s_h+1;
    int new_image_w = (image_w + 2*p_w - k_w)/s_w+1;

    double* conv2Output = (double*)malloc(new_image_h*new_image_w*c_mid*sizeof(double));
    depthwise_convolution(c_mid, k_h, k_w, s_h, s_w, p_h, p_w, image_h, image_w, conv1Output, conv2_weights, conv2_bias, conv2Output);
    relu(conv2Output, conv2Output, new_image_h*new_image_w*c_mid);

    // Output Convolutions (Project)
    convolution(c_mid, c_out, 1, 1, 1, 1, 0, 0, new_image_h, new_image_w, conv2Output, conv3_weights, conv3_bias, output);
    
    free(conv1Output);
    free(conv2Output);
}



void MobileNetV3_Small(double flattenedImage[], int imageWidth, int imageHeight, int imageDepth, double output[]) {

    // Assuming Image Size is 224x224x3
    // Layer 1
    double* conv1Output = new double[112*112*16];
    convolution(3, 16, 3, 3, 2, 2, 1, 1, 224, 224, flattenedImage, Conv_weights, Conv_bias, conv1Output);
    hard_swish(conv1Output, conv1Output, 112*112*16);


    // Block1
    double *conv2Output = new double[56*56*16];
    block_0(16, 16, 16, 3, 3, 2, 2, 1, 1, 112, 112, conv1Output, expanded_conv_depthwise_weights, expanded_conv_depthwise_bias, expanded_conv_squeeze_excite_1_weights, expanded_conv_squeeze_excite_1_bias, expanded_conv_squeeze_excite_2_weights, expanded_conv_squeeze_excite_2_bias, expanded_conv_project_weights, expanded_conv_project_bias, conv2Output);
    free(conv1Output);

    // Block2
    double* conv3Output = new double[28*28*24];
    block_2(16, 24, 72, 3, 3, 2, 2, 1, 1, 56, 56, conv2Output, conv3Output, expanded_conv_1_expand_weights, expanded_conv_1_expand_bias, expanded_conv_1_depthwise_weights, expanded_conv_1_depthwise_bias, expanded_conv_1_project_weights, expanded_conv_1_project_bias);
    free(conv2Output);

    // Block3
    double* conv4Output = new double[28*28*24];
    block_2(24, 24, 88, 3, 3, 1, 1, 1, 1, 28, 28, conv3Output, conv4Output, expanded_conv_2_expand_weights, expanded_conv_2_expand_bias, expanded_conv_2_depthwise_weights, expanded_conv_2_depthwise_bias, expanded_conv_2_project_weights, expanded_conv_2_project_bias);

    // Residual Connection
    add(conv3Output, conv4Output, conv4Output, 28*28*24);
    free(conv3Output);

    // Block4
    double* conv5Output = new double[14*14*40];
    block_1(24, 40, 96, 24, 5, 5, 2, 2, 2, 2, 28, 28, conv4Output, expanded_conv_3_expand_weights, expanded_conv_3_expand_bias, expanded_conv_3_depthwise_weights, expanded_conv_3_depthwise_bias, expanded_conv_3_squeeze_excite_1_weights, expanded_conv_3_squeeze_excite_1_bias, expanded_conv_3_squeeze_excite_2_weights, expanded_conv_3_squeeze_excite_2_bias, expanded_conv_3_project_weights, expanded_conv_3_project_bias, conv5Output);
    free(conv4Output);

    // Block5
    double* conv6Output = new double[14*14*40];
    block_1(40, 40, 240, 64, 5, 5, 1, 1, 2, 2, 14, 14, conv5Output, expanded_conv_4_expand_weights, expanded_conv_4_expand_bias, expanded_conv_4_depthwise_weights, expanded_conv_4_depthwise_bias, expanded_conv_4_squeeze_excite_1_weights, expanded_conv_4_squeeze_excite_1_bias, expanded_conv_4_squeeze_excite_2_weights, expanded_conv_4_squeeze_excite_2_bias, expanded_conv_4_project_weights, expanded_conv_4_project_bias, conv6Output);
    free(conv5Output);

    // Block6
    double* conv7Output = new double[14*14*40];
    block_1(40, 40, 240, 64, 5, 5, 1, 1, 2, 2, 14, 14, conv6Output, expanded_conv_5_expand_weights, expanded_conv_5_expand_bias, expanded_conv_5_depthwise_weights, expanded_conv_5_depthwise_bias, expanded_conv_5_squeeze_excite_1_weights, expanded_conv_5_squeeze_excite_1_bias, expanded_conv_5_squeeze_excite_2_weights, expanded_conv_5_squeeze_excite_2_bias, expanded_conv_5_project_weights, expanded_conv_5_project_bias, conv7Output);
    free(conv6Output);

    // Block7
    double* conv8Output = new double[14*14*48];
    block_1(40, 48, 120, 32, 5, 5, 1, 1, 2, 2, 14, 14, conv7Output, expanded_conv_6_expand_weights, expanded_conv_6_expand_bias, expanded_conv_6_depthwise_weights, expanded_conv_6_depthwise_bias, expanded_conv_6_squeeze_excite_1_weights, expanded_conv_6_squeeze_excite_1_bias, expanded_conv_6_squeeze_excite_2_weights, expanded_conv_6_squeeze_excite_2_bias, expanded_conv_6_project_weights, expanded_conv_6_project_bias, conv8Output);
    free(conv7Output);

    // Block8
    double* conv9Output = new double[14*14*48];
    block_1(48, 48, 144, 40, 5, 5, 1, 1, 2, 2, 14, 14, conv8Output, expanded_conv_7_expand_weights, expanded_conv_7_expand_bias, expanded_conv_7_depthwise_weights, expanded_conv_7_depthwise_bias, expanded_conv_7_squeeze_excite_1_weights, expanded_conv_7_squeeze_excite_1_bias, expanded_conv_7_squeeze_excite_2_weights, expanded_conv_7_squeeze_excite_2_bias, expanded_conv_7_project_weights, expanded_conv_7_project_bias, conv9Output);
    free(conv8Output);

    // Block9
    double* conv10Output = new double[7*7*96];
    block_1(48, 96, 288, 72, 5, 5, 2, 2, 2, 2, 14, 14, conv9Output, expanded_conv_8_expand_weights, expanded_conv_8_expand_bias, expanded_conv_8_depthwise_weights, expanded_conv_8_depthwise_bias, expanded_conv_8_squeeze_excite_1_weights, expanded_conv_8_squeeze_excite_1_bias, expanded_conv_8_squeeze_excite_2_weights, expanded_conv_8_squeeze_excite_2_bias, expanded_conv_8_project_weights, expanded_conv_8_project_bias, conv10Output);
    free(conv9Output);

    // Block10
    double* conv11Output = new double[7*7*96];
    block_1(96, 96, 576, 144, 5, 5, 1, 1, 2, 2, 7, 7, conv10Output, expanded_conv_9_expand_weights, expanded_conv_9_expand_bias, expanded_conv_9_depthwise_weights, expanded_conv_9_depthwise_bias, expanded_conv_9_squeeze_excite_1_weights, expanded_conv_9_squeeze_excite_1_bias, expanded_conv_9_squeeze_excite_2_weights, expanded_conv_9_squeeze_excite_2_bias, expanded_conv_9_project_weights, expanded_conv_9_project_bias, conv11Output);
    free(conv10Output);

    // Block11
    double* conv12Output = new double[7*7*96];
    block_1(96, 96, 576, 144, 5, 5, 1, 1, 2, 2, 7, 7, conv11Output, expanded_conv_10_expand_weights, expanded_conv_10_expand_bias, expanded_conv_10_depthwise_weights, expanded_conv_10_depthwise_bias, expanded_conv_10_squeeze_excite_1_weights, expanded_conv_10_squeeze_excite_1_bias, expanded_conv_10_squeeze_excite_2_weights, expanded_conv_10_squeeze_excite_2_bias, expanded_conv_10_project_weights, expanded_conv_10_project_bias, conv12Output);
    free(conv11Output);

    // Layer 12
    double* conv13Output = new double[7*7*576];
    convolution(96, 576, 1, 1, 1, 1, 0, 0, 7, 7, conv12Output, Conv_1_weights, Conv_1_bias, conv13Output);
    hard_swish(conv13Output, conv13Output, 7*7*576);
    free(conv12Output);

    // Layer 13
    double* conv14Output = new double[576];
    avgPooling(conv13Output, conv14Output, 7, 7, 576, 7, 7, 1, 1, 0, 0);
    free(conv13Output);

    // Layer 14
    double* conv15Output = new double[1024];
    convolution(576, 1024, 1, 1, 1, 1, 0, 0, 1, 1, conv14Output, Conv_2_weights, Conv_2_bias, conv15Output);
    hard_swish(conv15Output, conv15Output, 1024);
    free(conv14Output);

    // Layer 15
    double* conv16Output = new double[1000];
    convolution(1024, 1000, 1, 1, 1, 1, 0, 0, 1, 1, conv15Output, Logits_weights, Logits_bias, conv16Output);
    free(conv15Output);

    // // Softmax
    // softmax(conv16Output, output, 1000);
    // free(conv16Output);
}

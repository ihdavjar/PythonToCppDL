// MODEL_H
#ifndef MODEL_H
#define MODEL_H

void block_0(int c_in, int c_out, int c_squeeze, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int image_h, int image_w, double flattenedImage[], const double conv1_weights[], const double conv1_bias[], const double conv2_weights[], const double conv2_bias[], const double conv3_weights[], const double conv3_bias[], const double conv4_weights[], const double conv4_bias[], double output[]);
void block_1(int c_in, int c_out, int c_mid, int c_squeeze, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int image_h, int image_w, double flattenedImage[], const double conv1_weights[], const double conv1_bias[], const double conv2_weights[], const double conv2_bias[], const double conv3_weights[], const double conv3_bias[], const double conv4_weights[], const double conv4_bias[], const double conv5_weights[], const double conv5_bias[], double output[]);
void block_2(int c_in, int c_out, int c_mid, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int image_h, int image_w, double flattenedImage[], double output[], const double conv1_weights[], const double conv1_bias[], const double conv2_weights[], const double conv2_bias[], const double conv3_weights[], const double conv3_bias[]);
void MobileNetV3_Small(double flattenedImage[], int imageWidth, int imageHeight, int imageDepth, double output[]);


#endif
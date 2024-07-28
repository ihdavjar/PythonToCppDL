// MODEL_H
#ifndef MODEL_H
#define MODEL_H

void block_2(int c_in, int c_out, double input[], double output[], const double conv1_conv_weights[], const double conv1_conv_bias[], const double conv2_conv_weights[], const double conv2_conv_bias[]);
void block_3(int c_in, int c_out, int stride,double input[], double output[], const double conv1_conv_weights[], const double conv1_conv_bias[], const double conv2_conv_weights[], const double conv2_conv_bias[], const double conv3_conv_weights[], const double conv3_conv_bias[]);
void block_4(int c_in, int c_out, int stride, double input[], double output[], const double conv1_conv_weights[], const double conv1_conv_bias[], const double conv2_conv_weights[], const double conv2_conv_bias[], const double conv3_conv_weights[], const double conv3_conv_bias[]);
void block_5(int c_in, int c_out, int stride, double input[], double output[], const double conv1_conv_weights[], const double conv1_conv_bias[], const double conv2_conv_weights[], const double conv2_conv_bias[], const double conv3_conv_weights[], const double conv3_conv_bias[]);
void ResNet50(double flattenedImage[], int imageWidth, int imageHeight, int imageDepth, double output[]);


#endif
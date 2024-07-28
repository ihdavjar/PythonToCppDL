// layers.cpp
#ifndef LAYERS_H
#define LAYERS_H

// Activation function
void relu(double input[], double output[], int size);
void leaky_relu(double input[], double output[], int size);
void sigmoid(double input[], double output[], int size);
void tanh(double input[], double output[], int size);
void softmax(double input[], double output[], int size);
void hard_swish(double input[], double output[], int size);

// Padding
void pad_image(double input[], double output[], int imageWidth, int imageHeight, int imageDepth, int p_h, int p_w);
void add_padding_middle(double input[], double output[], int imageWidth, int imageHeight, int imageDepth, int z_h, int z_w);

// Convolution
void convolution (int c_in, int c_out, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int image_h, int image_w, double flattenedImage[], const double kernels[], const double biases[], double output[]);
void transposed_convolution(int c_in, int c_out, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int image_h, int image_w, double flattenedImage[], const double kernels[], const double biases[], double output[]);

// Pooling
void maxPooling(double flattenedImage[], double output[], int image_h, int image_w, int numChannels, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w);
void avgPooling(double flattenedImage[], double output[], int image_h, int image_w, int numChannels, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w);

// Fully connected
void linear(double input[], double output[], const double weights[], const double bias[], int inputSize, int outputSize);

// Noise
void addNoise(double input[], double stddev, int input_size);

// Misc
void add(double input1[], double input2[], double output[], int size);

#endif // LAYERS_H
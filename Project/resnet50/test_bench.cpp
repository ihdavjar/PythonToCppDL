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

#include "test_image.h"
#include "model.h"
#include "test_data_class0.h"
#include "test_data_class1.h"

using namespace std;

int main() {
    int output_final[2];
    
    // Number of image samples
    for(int k=0;k<2;k++){
        double* flattenedImage = new double[224*224*3];
        int idx = 0;
        int ido=0;

        for(int d=0;d<3;d++){
            for (int i = 0; i < 224; ++i) {
                for (int j = 0; j < 224; ++j) {
                    if (k==0){
                    flattenedImage[idx++] = test_image_class0[ido++];
                    }
                    else{
                    flattenedImage[idx++] = test_image_class1[ido++];
                    }
                }
            }
        }

        double* output = new double[2];  // Assuming the output size is 10
            
        ResNet50(flattenedImage, 224, 224, 3, output);
        

        auto maxElementIterator = std::max_element(output, output + 2);
        int maxIndex = std::distance(output, maxElementIterator);
        output_final[k]=maxIndex;
        
        cout<<output_final[k]<<endl;
    }

    return 0;
}

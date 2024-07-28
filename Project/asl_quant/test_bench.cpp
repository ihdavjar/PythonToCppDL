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


int main() {
    int output_final[1];
    int ido=0;
    for(int k=0;k<1;k++){
        double flattenedImage[64*64*3];
        int idx = 0;
        for(int d=0;d<3;d++){
            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j < 64; ++j) {
                    flattenedImage[idx++] = test_image[ido++];
                }
            }
        }
        
        double output[29];  // Assuming the output size is 10
        asl_quant(flattenedImage, 64,64,3,output);
        auto maxElementIterator = std::max_element(output, output + 10);
        int maxIndex = std::distance(output, maxElementIterator);
        output_final[k]=maxIndex;
    }
    // int count =0;
    // for(int i=0;i<100;i++){
    //     // cout<if (len(splitted_key)>3):
        cout<<output_final[0]<<endl;
    //     if(output_final[i]!=test_prediction[i]){
    //         count++;
    //     }
    // }
    // cout<<count<<endl;
    return 0;
}

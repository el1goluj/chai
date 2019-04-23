/*
 * Copyright (c) 2018 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include <unistd.h>
#include <atomic>
#include <iostream>
#include <fstream>
using namespace std;


void read_input(float* input_image, int size) {

        FILE *fp = fopen("../input/image.txt", "r");
        if(fp == NULL)
            exit(EXIT_FAILURE);
        for(int i = 0; i < size; i++) {
                fscanf(fp, "%f ", &input_image[i]);
        }
        fclose(fp);
}

void read_weights(float* input_weights, int size) {

        FILE *fp = fopen("../input/weights.txt", "r");
        if(fp == NULL)
            exit(EXIT_FAILURE);
        for(int i = 0; i < size; i++) {
                fscanf(fp, "%f ", &input_weights[i]);
        }
        fclose(fp);
}

int main(int argc, char **argv) {

	#define B 10000
	#define M 50
	#define C 1
	#define H 28
	#define W 28
	#define K 10

	int H_out = H - K + 1;
	int W_out = W - K + 1;

	size_t weight_size = M*C*K*K;
    float* input_weights = (float *) malloc(weight_size * sizeof(float));
    read_weights(input_weights, weight_size);


    int image_size = B * C * H * W;
    float* input_image = (float *) malloc(image_size * sizeof(float));
    read_input(input_image, image_size);

    double sum = 0.0f;
    double temp = 0.0f;

    for(int b = 0; b < B; ++b) {
        for(int m = 0; m < M; m++) {
            for(int h = 0; h < H_out; h++) {
                for(int w = 0; w <W_out; w++) {
                    temp = 0.0f;
                     for(int c = 0; c < C; c++) {
                        for(int p = 0; p < K; p++) {
                            for(int q = 0; q < K; q++) {
                                temp += (double) (input_image[b*(C * H * W) + c * (H * W) + (h + p) * W + (w + q)] * input_weights[m * (C * K * K) + c * (K * K) + p * K + q]);
                            }
                        }
                    }
                    if(temp < 0.0f)
                        temp = 0.0f;
                    sum += temp;
               }
            }
        }
    }

    // Write the correct output to a file. 
    FILE *out_file = fopen("expected_output.txt", "w");
    if(!out_file) {
        printf("Error Reading output file\n");
        return 1;
    }  
    fprintf(out_file, "%lf", sum);
    fclose(out_file);
  	return 0;
}
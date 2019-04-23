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

#include "kernel.h"
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>

void run_cpu_unroll_weights (int num_threads, float *w, float *w_unroll, int C, int K, int M) {
    
    #define k4d(i3,i2,i1,i0) w[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]
    std::vector<std::thread> cpu_threads_unroll;
    for(int k = 0; k < num_threads; k++) {
        cpu_threads_unroll.push_back(std::thread([=]() {
            int c,m,w_row,w_col;
            for(m = 0; m < M; m++)  {
               for(c = 0; c < C; c++){
                    for(w_row = 0; w_row < K; w_row++){
                        for(w_col = k; w_col < K; w_col+= num_threads)  {
                            w_unroll[C * K * K * m + c * K * K + w_row * K + w_col] = k4d(m, c, w_row, w_col);
                        }
                    }
                }
            }
        }));
    }
    std::for_each(cpu_threads_unroll.begin(), cpu_threads_unroll.end(), [](std::thread &t) { t.join(); });
    #undef k4d
}

void run_cpu_cnn (int num_threads, const float* input_image, const float* input_weights, float* output,
                  const int B, const int M, const int H, const int W, const int K,
                  const int C, const int H_out, const int W_out ) {
    
    #define k4d(i3,i2,i1,i0) input_weights[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]
    #define o4d(i3,i2,i1,i0) output[i3 * (H_out * W_out * M ) + i2 * (W_out * M ) + i1 * M + i0]
    std::vector<std::thread> cpu_threads_cnn;

    for(int k = 0; k < num_threads; k++) {
        cpu_threads_cnn.push_back(std::thread([=]() {
            double temp = 0.0f;
            for(int b = 0; b < B; ++b) {
                for(int m = 0; m < M; m++) {
                    for(int h = 0; h < H_out; h++) {
                        for(int w = k; w <W_out; w+= num_threads) {
                            temp = 0.0f;
                             for(int c = 0; c < C; c++) {
                                for(int p = 0; p < K; p++) {
                                    for(int q = 0; q < K; q++) {
                                        temp += (double) (input_image[b*(C * H * W) + c * (H * W) + (h + p) * W + (w + q)] * k4d(m, c, p, q));
                                    }
                                }
                            }
                            if(temp < 0.0f)
                                temp = 0.0f;
                            o4d(b,h,w,m)= temp;
                       }
                    }
                }
            }
        }));
    }
    std::for_each(cpu_threads_cnn.begin(), cpu_threads_cnn.end(), [](std::thread &t) { t.join(); });
    #undef o4d
    #undef k4d
}





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

#include "cuda_runtime.h"
#include "support/common.h"
#include <vector>
#include <algorithm>
#include <string.h>

void run_cpu_unroll_weights (int num_threads, float *w, float *w_unroll, int C, int K, int M);

void run_cpu_cnn (int num_threads, const float* input_image, const float* input_weights, float* output,
                  const int B, const int M, const int H, const int W, const int K,
                  const int C, const int H_out, const int W_out );

cudaError_t call_unroll_weights(float *w, float *w_unroll, const int C, const int M, const int H, 
								const int W, const int K);

cudaError_t call_unroll_inputs(const float *x, float *x_unroll, const int B, const int C, const int H, 
							   const int W, const int K, const int H_out, const int W_out);

cudaError_t call_matrix_multiply(float *A, float *B, float *y, 
								 int numARows, int numAColumns, int numBRows, int numBColumns,
								 int numCRows, int numCColumns, const int M, const int H_out, 
								 const int W_out, const int C, const int K, const int b);

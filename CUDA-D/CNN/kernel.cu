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

#include "support/common.h"
#define TILE_WIDTH 32

 __global__ void unroll_weights(float *w, float *w_unroll, const int C, const int M, const int H, const int W, const int K) {
       #define k4d(i3,i2,i1,i0) w[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]
       int c, m, w_row, w_col;
        for(m = 0; m < M; m++)  {
           for(c = 0; c < C; c++){
                for(w_row = 0; w_row < K; w_row++){
                    for(w_col = 0; w_col < K; w_col++)  {
                        w_unroll[C * K * K * m + c * K * K + w_row * K + w_col] = k4d(m, c, w_row, w_col);
                    }
                }
            }
         }
       #undef k4d
}

__global__ void unroll_inputs(const float *x, float *x_unroll, const int C, const int H, const int W, const int K) {
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int num = blockIdx.z;
    int offset = num *  C * K * K * H_out * W_out;
    int W_unroll = H_out * W_out; // width of unrolled matrix
    int w_unroll;
    int h_base;
    int h_unroll;
    int c, s, h_out, w_out, p ,q;
    if(t < C * W_unroll) { //change the size of grid and block 
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out; // row #                   
        w_out = s % W_out; // col #
        w_unroll = h_out * W_out + w_out; // col index
        h_base = c * K * K;
        for( p = 0; p < K; p++) {
            for( q = 0; q < K; q++) {
                h_unroll = h_base + p * K + q; // row index
                x_unroll[h_unroll * W_unroll + w_unroll + offset] = x4d(num, c, h_out + p, w_out + q);
            }
        }
    }
    #undef x4d
}

__global__ void matrixMultiply (float *A, float *B, float *y, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns,
                               const int M, int H_out, int W_out, int C, int K) {
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ unsigned char Nds[TILE_WIDTH][TILE_WIDTH];
  int num = blockIdx.z;
  int offset = num * M * H_out * W_out;
  int offset_input = num *  C * K * K * H_out * W_out;
  int Width = numAColumns;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;

  for (int ph = 0; ph < ceil((float) Width/TILE_WIDTH); ++ph) {
    if((Row < numCRows) && (ph*TILE_WIDTH + tx) < numAColumns) {
      Mds[ty][tx] = A[Row * numAColumns + ph * TILE_WIDTH + tx];
    }
    else {
      Mds[ty][tx] = 0;
    }

    if((ph * TILE_WIDTH+ty) < numAColumns && Col < numBColumns) {
      Nds[ty][tx] = B[(ph*TILE_WIDTH + ty)* numCColumns + Col + offset_input];
    }
    else {
      Nds[ty][tx] = 0;
    }
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue +=  Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }

  if ((Row < numARows) && (Col < numBColumns)) {
    if(Pvalue < 0){
        y[offset + Row * numCColumns + Col] = 0;
    }
    else{
        y[offset + Row * numCColumns + Col] = Pvalue;
    }
  }
 }

cudaError_t call_unroll_weights(float *w, float *w_unroll, const int C, const int M, const int H, const int W, const int K){
    dim3 dimGrid(1,1,1);
    dim3 dimBlock(1,1,1);
    unroll_weights<<<dimGrid, dimBlock>>>(w, w_unroll, C, M, H, W, K);
    cudaError_t err = cudaGetLastError();
    return err;
}

cudaError_t call_unroll_inputs(const float *x, float *x_unroll, const int B, const int C, const int H, const int W, const int K, const int H_out, const int W_out) {
    dim3 dimBlock(H_out * W_out, 1, 1);
    dim3 dimGrid(C, 1, B);
    unroll_inputs<<<dimGrid, dimBlock>>>(x, x_unroll, C, H, W, K);
    cudaError_t err = cudaGetLastError();
    return err;
}

cudaError_t call_matrix_multiply(float *A, float *B, float *y, int numARows,
                               int numAColumns, int numBRows, int numBColumns, int numCRows,
                               int numCColumns, const int M, const int H_out, const int W_out, const int C, const int K, const int b) {
    dim3 dimGrid(ceil((float) (H_out * W_out)/ TILE_WIDTH), ceil((float) M/TILE_WIDTH), b);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMultiply<<<dimGrid, dimBlock>>>(A, B, y, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns, M, H_out, W_out, C, K);
    cudaError_t err = cudaGetLastError();
    return err;
}

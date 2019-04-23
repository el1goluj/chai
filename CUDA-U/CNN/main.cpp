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

#include "support/cuda-setup.h"
#include "kernel.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"

#include <unistd.h>
#include <thread>
#include <atomic>
#include <assert.h>

// Params ---------------------------------------------------------------------
struct Params {

    int         device;
    int         tile_width;
    int         n_gpu_threads;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    const char *input_file_name;
    const char *weights_file_name;
    const char *output_comparison_file;
    int         display = 0;

    Params(int argc, char **argv) {
        device          = 0;
        n_gpu_threads   = 32;
        n_threads       = 8;
        n_warmup        = 5;
        n_reps          = 95;
        input_file_name = "./input/image.txt";
        weights_file_name = "./input/weights.txt";
        output_comparison_file = "./output/expected_output.txt";
        int opt;
        while((opt = getopt(argc, argv, "h:d:i:t:w:r:f:q:c:x")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'd': device          = atoi(optarg); break;
            case 'i': n_gpu_threads   = atoi(optarg); break;
            case 't': n_threads       = atoi(optarg); break;
            case 'w': n_warmup        = atoi(optarg); break;
            case 'r': n_reps          = atoi(optarg); break;
            case 'f': input_file_name = optarg; break;
            case 'q': weights_file_name = optarg; break;
            case 'c': output_comparison_file = optarg; break;
            case 'x': display         = 1; break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        assert(display == 0 && "Video output with CHAI_OPENCV not supported");
        assert((n_gpu_threads == 32) || (n_gpu_threads == 0) && "Number of device threads per block must be 32 or 0!");
        assert(n_threads >= 0 && "Invalid # of host threads!");
        assert((n_threads > 0) || (n_gpu_threads > 0) && "Must use either (or both) CPU or GPU!");
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./cedt [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -d <D>    CUDA device ID (default=0)"
                "\n    -i <I>    # of device threads per block. set to 0 for CPU-only operation (default=32)"
                "\n    -t <T>    # of host threads. set to 0 for GPU-only operation (default=8)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=95)"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    file containing input data (default = ./input/image.txt)"
                "\n    -q <Q>    file containing weights data (default = ./input/weights.txt)"
                "\n    -c <C>    file containing output comparison data (default=./output/expected_output.txt)"
                "\n    -x        display output video (with CHAI_OPENCV)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input(float* input_image, int size, const Params &p) {

        FILE *fp = fopen(p.input_file_name, "r");
        if(fp == NULL)
            exit(EXIT_FAILURE);
        for(int i = 0; i < size; i++) {
                fscanf(fp, "%f ", &input_image[i]);
        }
        fclose(fp);
}

void read_weights(float* input_weights, int size, const Params &p) {

        FILE *fp = fopen(p.weights_file_name, "r");
        if(fp == NULL)
            exit(EXIT_FAILURE);
        for(int i = 0; i < size; i++) {
                fscanf(fp, "%f ", &input_weights[i]);
        }
        fclose(fp);
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    Params       p(argc, argv);
    CUDASetup    setcuda(p.device);
    if(p.n_gpu_threads == 0) {
        printf("\nUsing CPU Only Configuration\n");
    }
    else if (p.n_threads == 0) {
        printf("\nUsing GPU Only Configuration\n");
    }
    else {
        printf("\nUsing Collaborative CPU/GPU Configuration\n");
    }
    Timer        timer;
    cudaError_t  cudaStatus;

    /* Initialize - read in file data  */
    timer.start("Initialization");

    const int max_gpu_threads = setcuda.max_gpu_threads();
    
    // Define threads IDs for proxy threads 
    const int GPU_PROXY = 1; // We want GPU thread to run first 
    const int CPU_PROXY = 0;
    const int num_gpu_reps = 5;
    int cpu_rep;

    // Input/Output file parameters
    const int B = 10000; // Number of batches
    const int C = 1; // Number of input feature maps
    const int H = 28; // Height of input map
    const int W = 28; // Width of input map
    const int K = 10; // Mask width
    const int M = 50; // Number of output feature maps
    int H_out = H - K + 1; // Height of output map
    int W_out = W - K + 1; // Width of output map

    // Capture weights data 
    size_t weight_size = M*C*K*K;
    //float* input_weights = (float *) malloc(weight_size * sizeof(float));
    float * input_weights;
    cudaStatus = cudaMallocManaged(&input_weights, weight_size * sizeof(float));
    CUDA_ERR();
    read_weights(input_weights, weight_size, p);
    weight_size *= sizeof(float);

    // Capture input data
    int image_size = B * C * H * W;
    float* input_image = (float *) malloc(image_size * sizeof(float));
    ALLOC_ERR(input_image);
    read_input(input_image, image_size, p);    
    timer.stop("Initialization");
    timer.print("Initialization", 1);

    /* Allocate local buffers */
    timer.start("Allocation");
    
    // Begin host data allocation
    int num_threads = p.n_warmup + p.n_reps;
    size_t in_size = (B/num_threads) * C * H * W * sizeof(float);

    // When using GPU, localize host input data into thread-friendly subunits 
    //float ** h_input = (float **)malloc((num_threads) * sizeof(float *));
    float ** h_input;
    cudaStatus = cudaMallocManaged(&h_input, (num_threads) * sizeof(float *));
    CUDA_ERR();
    for (int idx=0; idx<num_threads; idx++) {
        //h_input[idx] = (float *) malloc(in_size);
        cudaStatus = cudaMallocManaged(&(h_input[idx]), in_size);
        CUDA_ERR();
        memcpy(h_input[idx], input_image +(B/num_threads) * C * H * W * idx, in_size);
    }
    
    // Allocate host weights
    float *h_weights_unrolled;
    //h_weights_unrolled = (float *) malloc(weight_size);
    cudaStatus = cudaMallocManaged(&h_weights_unrolled, weight_size);
    CUDA_ERR();  

    // Allocate host output data
    size_t out_size = (B/num_threads) * H_out * W_out * M * sizeof(float);

    // CPU Only
    float * h_output_cpu= (float *)malloc((num_threads) * out_size);
    ALLOC_ERR(h_output_cpu);
    float ** h_final = (float **)malloc((num_threads) * sizeof(float *)); 
    ALLOC_ERR(h_final);
    for(int i = 0; i < num_threads; i++) {
        h_final[i] = (float *)malloc(out_size);
        ALLOC_ERR(h_final[i]);
    } 

    //float ** h_output= (float **)malloc((num_threads) * sizeof(float *)); 
    float ** h_output;
    cudaStatus = cudaMallocManaged(&h_output, (num_threads) * sizeof(float *));
    CUDA_ERR();
    for(int i = 0; i < num_threads; i++) {
        //h_output[i] = (float *)malloc(out_size);
        cudaStatus = cudaMallocManaged(&(h_output[i]), out_size);
        CUDA_ERR();
    }

    // Allocate GPU input and unrolled input data
    //float *h_input_unrolled[num_gpu_reps];
    float ** h_input_unrolled;
    cudaStatus = cudaMallocManaged(&h_input_unrolled, num_gpu_reps * sizeof(float *));
    CUDA_ERR();
    size_t input_unrolled_size = (B/num_threads)*C*K*K*H_out*W_out*sizeof(float);
    for(int i=0; i<num_gpu_reps; i++) {
        cudaStatus = cudaMallocManaged(&(h_input_unrolled[i]), input_unrolled_size);
        CUDA_ERR();
    } 


    /*
    float *d_input[num_gpu_reps], *d_input_unrolled[num_gpu_reps];
    size_t input_unrolled_size = (B/num_threads)*C*K*K*H_out*W_out*sizeof(float);
    for(int i=0; i<num_gpu_reps; i++) {
        cudaStatus = cudaMalloc((void**)&d_input[i], in_size);   
        CUDA_ERR();
        cudaStatus = cudaMalloc((void**)&d_input_unrolled[i], input_unrolled_size);
        CUDA_ERR();
    } 

    // Allocate GPU weights and unrolled weights data
    float * d_weights_unrolled;
    cudaStatus = cudaMalloc((void**)&d_weights_unrolled, weight_size);
    CUDA_ERR();

    float * d_weights;
    cudaStatus = cudaMalloc((void**)&d_weights, weight_size);
    CUDA_ERR();

    // Allocate GPU output data 
    float * d_output[num_gpu_reps];
    for(int i=0; i<num_gpu_reps; i++) {
        cudaStatus = cudaMalloc((void**)&d_output[i], out_size);
        CUDA_ERR(); 
    }

    // Synchronize across CUDA devices
    cudaDeviceSynchronize();
    CUDA_ERR(); */

    // Initialize thread ready signals  
    std::atomic<int> cpu_ready[num_threads/num_gpu_reps];    
    for(int i = 0; i < num_threads/num_gpu_reps; i++) {
        cpu_ready[i].store(0);
    }

    timer.stop("Allocation");
    timer.print("Allocation", 1);

    /* Begin using proxy threads */
    timer.start("Total Proxies");
    std::vector<std::thread> proxy_threads;

    // Iterate through both proxy threads 
    if (p.n_gpu_threads != 0) {
        for(int proxy_tid = 0; proxy_tid < 2; proxy_tid++) {
            proxy_threads.push_back(std::thread([&, proxy_tid]() {

                // Iterate through all available non-proxy threads
                for(int rep = 0; rep < num_threads/num_gpu_reps; rep++) {

                    // Current thread is GPU PROXY thread 
                    if(proxy_tid == GPU_PROXY) {
                        // Wait for GPU proxy to indicate it's done
                        while((&cpu_ready[rep])->load() == 0) {
                        }

                        // If GPU had invalid data then move on
                        if((&cpu_ready[rep])->load() == -1)
                            continue;    

                        /* Copy GPU input and weights data to device 
                        timer.start("GPU Proxy: Copy To Device");
                        for (int i=0; i<num_gpu_reps; i++) {
                            cudaStatus = cudaMemcpy(d_input[i], h_input[rep*num_gpu_reps+i], in_size, cudaMemcpyHostToDevice);
                            CUDA_ERR();
                        }
                        // Either copy unrolled weights, or input weights to GPU
                        if (p.n_threads > 0) {
                            cudaStatus = cudaMemcpy(d_weights_unrolled, h_weights_unrolled, weight_size, cudaMemcpyHostToDevice);
                            CUDA_ERR();
                        }
                        else {
                            cudaStatus = cudaMemcpy(d_weights, input_weights, weight_size, cudaMemcpyHostToDevice);
                            CUDA_ERR();
                        }
                        cudaDeviceSynchronize();

                        timer.stop("GPU Proxy: Copy To Device");*/

                        /* Actually launch GPU kernels */
                        timer.start("GPU Proxy: Kernel");
                        if (p.n_threads == 0) {
                            // Call unroll weights if in GPU only configuration
                            cudaStatus = call_unroll_weights(input_weights, h_weights_unrolled, C, M, H, W, K);
                            CUDA_ERR();
                        }
                        for (int i=0; i<num_gpu_reps; i++) {
                            // UNROLL INPUT KERNEL
                            cudaStatus = call_unroll_inputs(h_input[rep*num_gpu_reps + i], h_input_unrolled[i], (B/num_threads), C, H, W, K, H_out, W_out);
                            CUDA_ERR();    

                            //cudaDeviceSynchronize();///////?

                            // MATRIX MULTIPLY KERNEL
                            cudaStatus = call_matrix_multiply(h_weights_unrolled, h_input_unrolled[i], h_output[rep*num_gpu_reps + i], M, K*K*C, K*K*C,
                                                           H_out*W_out, M, H_out*W_out, M, H_out, W_out, C, K, (B/num_threads));
                            CUDA_ERR(); 
                        }

                        timer.stop("GPU Proxy: Kernel");

                        /* Copy GPU Data back to host and synchronize devices 
                        timer.start("GPU Proxy: Copy Back"); 
                        for (int i=0; i<num_gpu_reps; i++) {
                            cudaStatus = cudaMemcpy(h_output[rep*num_gpu_reps+i], d_output[i], out_size, cudaMemcpyDeviceToHost);
                            CUDA_ERR();
                            cudaDeviceSynchronize();
                        }
                        timer.stop("GPU Proxy: Copy Back"); */
                    } 

                    // Current thread is CPU Proxy thread
                    else if(proxy_tid == CPU_PROXY) {
                        // Move onto next non-proxy thread if input isn't ready 
                        if(input_weights == NULL) {
                            (&cpu_ready[rep])->store(-1);
                            continue;
                        }      

                        if (p.n_threads > 0) {
                            /* Launch CPU kernel function */
                            timer.start("CPU Proxy: Kernel");
                            std::thread main_thread(
                                run_cpu_unroll_weights, p.n_threads, input_weights, h_weights_unrolled, C, K, M);
                            main_thread.join();
                            timer.stop("CPU Proxy: Kernel");
                        }

                        // Release GPU proxy
                        (&cpu_ready[rep])->store(1);
                    }
                }
            }));
        }
        std::for_each(proxy_threads.begin(), proxy_threads.end(), [](std::thread &t) { t.join(); });
    }
    // CPU Only
    else {
        timer.start("CPU Proxy: Kernel");
        std::thread main_thread (
            run_cpu_cnn, p.n_threads, input_image, input_weights, h_output_cpu, B, M, H, W, K, C, H_out, W_out); 
        main_thread.join();
        timer.stop("CPU Proxy: Kernel");
    }

    // Synchronize CUDA devices 
    cudaDeviceSynchronize();
    CUDA_ERR();
    timer.stop("Total Proxies");

    // Print various timer reports 
    timer.print("Total Proxies", 1);
    printf("CPU Proxy:\n");
    printf("\t");
    timer.print("CPU Proxy: Kernel", 1);
    printf("GPU Proxy:\n");
    printf("\t");
    timer.print("GPU Proxy: Copy To Device", 1);
    printf("\t");
    timer.print("GPU Proxy: Kernel", 1);
    printf("\t");
    timer.print("GPU Proxy: Copy Back", 1);

    // Verify that our answer is correct
    // CPU only configuration
    if (p.n_gpu_threads == 0) {
        for(int i = 0; i < num_threads; i++) {
            memcpy(h_final[i], h_output_cpu + ((B/num_threads) * H_out * W_out * M) * i, out_size);
        } 
        verify(h_final, (B/num_threads) * H_out * W_out * M, p.output_comparison_file, num_threads);
    }
    // GPU only or collaborative configuration
    else {
        verify(h_output, (B/num_threads) * H_out * W_out * M, p.output_comparison_file, num_threads);
    }
    
    /* Begin deallocation process */
    timer.start("Deallocation");
    cudaStatus = cudaFree(input_weights);
    CUDA_ERR();
    free(input_image);
    for(int i = 0; i < num_threads; i++) {
        cudaStatus = cudaFree(h_input[i]);
        CUDA_ERR();
    }
    cudaStatus = cudaFree(h_input);
    CUDA_ERR();
    cudaStatus = cudaFree(h_weights_unrolled);
    CUDA_ERR();
    free(h_output_cpu);
    for(int i=0; i < num_threads; i++) {
        free(h_final[i]);
    }
    free(h_final);
    for(int i = 0; i < num_threads; i++) {
        cudaStatus = cudaFree(h_input_unrolled[i]);
        CUDA_ERR();
    }
    cudaStatus = cudaFree(h_input_unrolled);
    CUDA_ERR();

    for(int i=0; i < num_threads; i++) {
        cudaStatus = cudaFree(h_output[i]);
        CUDA_ERR();
    }
    cudaStatus = cudaFree(h_output);
    CUDA_ERR();
        
    // Device memory 
    /*for (int i=0; i<num_gpu_reps; i++) {
        cudaStatus = cudaFree(d_input[i]);
        cudaStatus = cudaFree(d_input_unrolled[i]);
        cudaStatus = cudaFree(d_output[i]);
    }
    cudaStatus = cudaFree(d_weights_unrolled);
    CUDA_ERR();*/

    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    // Release timers
    timer.release("Allocation");
    timer.release("Initialization");
    timer.release("Total Proxies");
    timer.release("CPU Proxy: Kernel");
    //timer.release("GPU Proxy: Copy To Device");
    timer.release("GPU Proxy: Kernel");
    //timer.release("GPU Proxy: Copy Back");
    timer.release("Deallocation");

    return 0;
}

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

#include "common.h"
#include <math.h>

inline int compare_output(float **output, int image_size, const char *file_name, int num_frames) {
    printf("Verification: \n");
    // Initialze sums to 0
    double calculated_sum = 0.0f;
    double expected_sum = 0.0f;
    double outPix = 0.0f;

    // Open output file 
    FILE *out_file = fopen(file_name, "r");
    if(!out_file) {
        printf("Error Reading output file\n");
        return 1;
    }
    else {
        printf("\tReading Expected Output: %s\n", file_name);
    }    
    fscanf(out_file, "%lf", &expected_sum);
    fclose(out_file);

    // Iterate through all the thread outputs  
    for(int i = 0; i < num_frames; i++) {

        // Iterate through each pixel
        for(int pixel = 0; pixel < image_size; pixel++) {

            // Read in our output pixel and increment the calculated sum
            outPix = (double) output[i][pixel];
	        calculated_sum += outPix; 
        }
    }

    // Make sure we are within allowed amounts of error (0.01%)
    printf("\tCalculated sum: %lf \n\tExpected sum: %lf \n", calculated_sum, expected_sum);
    float result = (float)expected_sum / (float)(calculated_sum);
    if((result > 1.0001f) || (result < 0.9999f)) {
        printf("\tVerification Test failed\n");
        exit(EXIT_FAILURE);
    }
    else {
        printf("\tVerification Test Passed\n");
    }
    return 0;
}

inline void verify(float **output, int image_size, const char *file_name, int num_frames) {
    compare_output(output, image_size, file_name, num_frames);
}

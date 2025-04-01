#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#define MAX_ITER 1000

__global__ void mandelbrotKernel(unsigned char* output, int width, int height, float x_min, float x_max, float y_min, float y_max);
void computeMandelbrot(unsigned char* d_output, int width, int height, float x_min, float x_max, float y_min, float y_max);
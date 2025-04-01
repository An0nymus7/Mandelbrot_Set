#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#define MAX_ITER 10000

__global__ void mandelbrotKernel(unsigned char* output, int width, int height, double x_min, double x_max, double y_min, double y_max);
void computeMandelbrot(unsigned char* d_output, int width, int height, double x_min, double x_max, double y_min, double y_max);
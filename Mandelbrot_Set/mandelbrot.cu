#include "mandelbrot.cuh"

__global__ void mandelbrotKernel(unsigned char* output, int width, int height, double x_min, double x_max, double y_min, double y_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= width || idy >= height) return;

    // Shared memory for z_real and z_imag (2 doubles per thread)
    extern __shared__ double shared_data[];
    double* z_real_shared = &shared_data[threadIdx.y * blockDim.x * 2 + threadIdx.x * 2];
    double* z_imag_shared = &shared_data[threadIdx.y * blockDim.x * 2 + threadIdx.x * 2 + 1];

    double real = x_min + (x_max - x_min) * idx / (double)width;
    double imag = y_min + (y_max - y_min) * idy / (double)height;
    *z_real_shared = 0;
    *z_imag_shared = 0;
    int iter = 0;

    while (iter < MAX_ITER && (*z_real_shared) * (*z_real_shared) + (*z_imag_shared) * (*z_imag_shared) <= 4.0) {
        double temp = (*z_real_shared) * (*z_real_shared) - (*z_imag_shared) * (*z_imag_shared) + real;
        *z_imag_shared = 2 * (*z_real_shared) * (*z_imag_shared) + imag;
        *z_real_shared = temp;
        iter++;
    }

    int pixel_idx = (idy * width + idx) * 4; // RGBA
    if (iter == MAX_ITER) {
        output[pixel_idx] = 0;     // R
        output[pixel_idx + 1] = 0; // G
        output[pixel_idx + 2] = 0; // B
        output[pixel_idx + 3] = 255; // A
    }
    else {
        output[pixel_idx] = (iter % 255);        // R
        output[pixel_idx + 1] = (iter * 2 % 255); // G
        output[pixel_idx + 2] = (iter * 3 % 255); // B
        output[pixel_idx + 3] = 255;             // A
    }
}

void computeMandelbrot(unsigned char* d_output, int width, int height, double x_min, double x_max, double y_min, double y_max) {
    dim3 blockSize(16, 16); // 256 threads/block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    size_t sharedMemSize = blockSize.x * blockSize.y * 2 * sizeof(double); // 4 KB for 16x16 block
    mandelbrotKernel << <gridSize, blockSize, sharedMemSize >> > (d_output, width, height, x_min, x_max, y_min, y_max);
    cudaDeviceSynchronize();
}
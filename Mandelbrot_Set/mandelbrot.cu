#include "mandelbrot.cuh"


__global__ void mandelbrotKernel(unsigned char* output, int width, int height, float x_min, float x_max, float y_min, float y_max, int max_iter)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= width || idy >= height) return;

	float real = x_min + (x_max - x_min) * idx / (float)width;
	float imag = y_min + (y_max - y_min) * idy / (float)height;
	float z_real = 0, z_imag = 0;
	int i = 0;

//try out without the second part of the condition in the while
	while (i < max_iter && z_real * z_real + z_imag * z_imag <= 4.0f)
	{
		float tmp = z_real * z_real - z_imag * z_imag + real;
		z_imag = 2 * z_real * z_imag + imag;
		z_real = tmp;
		i++;
	}

	int pixel_idx = (idy * width + idx) * 4;
	if (i == max_iter) {
		output[pixel_idx] = 0;			//R
		output[pixel_idx + 1] = 0;		//G
		output[pixel_idx + 2] = 0;		//B
		output[pixel_idx + 3] = 255;	// A 
	}
	else
	{
		output[pixel_idx] = (i % 255);			//R
		output[pixel_idx + 1] = (i * 2 % 255);	//G
		output[pixel_idx + 2] = (i * 3 % 255);	//B
		output[pixel_idx + 3] = 255;			// A
	}

}

void computeMandelbrot(unsigned char* d_output, int width, int height, float x_min, float x_max, float y_min, float y_max)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	mandelbrotKernel << <gridSize, blockSize >> > (d_output, width, height, x_min, x_max, y_min, y_max, MAX_ITER);
	cudaDeviceSynchronize();
}

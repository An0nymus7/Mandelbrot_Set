#include <cuda_runtime.h>
#include <iostream>

#include "mandelbrot.cuh"
#include "utils.h"


int main()
{
	float x_min = -2.0f, x_max = 1.0f;
	float y_min = -1.5f, y_max = 1.5f;
	bool needsUpdate = true;

	unsigned char* d_output;
	unsigned char* h_output = new unsigned char[WIDTH * HEIGHT * 4];
	if (!h_output) {
		std::cerr << "Failed to allocate h_output" << std::endl;
		return 1;
	}

	cudaMalloc(&d_output, WIDTH * HEIGHT * 4 * sizeof(unsigned char)); // 4 for RGBA

	sf::RenderWindow window;
	sf::Texture texture;
	sf::Sprite sprite;
	initWindow(window, texture, sprite);

	while (window.isOpen()) {
		handleEvents(window, x_min, x_max, y_min, y_max, needsUpdate);

		if (needsUpdate) {
			computeMandelbrot(d_output, WIDTH, HEIGHT, x_min, x_max, y_min, y_max);
			cudaError_t err= cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA memcpy FAILED: " << cudaGetErrorString(err);
				return 1;
			}
			updateTexture(texture, h_output);
			needsUpdate = false;
		}

		window.clear();
		window.draw(sprite);
		window.display();
	}
	cudaFree(d_output);
	delete[] h_output;
	return 0;
}
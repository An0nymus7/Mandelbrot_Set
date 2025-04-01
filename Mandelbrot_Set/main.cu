#include "mandelbrot.cuh"
#include "utils.h"
#include <cuda_runtime.h>
#include <iostream>

int main() {
    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;
    bool needsUpdate = true;
    sf::Vector2i lastMousePos;
    bool isDragging = false;

    unsigned char* d_output;
    unsigned char* h_output = new unsigned char[WIDTH * HEIGHT * 4];
    if (!h_output) {
        std::cerr << "Failed to allocate h_output" << std::endl;
        return 1;
    }
    cudaError_t err = cudaMalloc(&d_output, WIDTH * HEIGHT * 4 * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        delete[] h_output;
        return 1;
    }

    sf::RenderWindow window;
    sf::Texture texture;
    sf::Sprite sprite;
    initWindow(window, texture, sprite);

    // Initial render
    computeMandelbrot(d_output, WIDTH, HEIGHT, x_min, x_max, y_min, y_max);
    cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
    updateTexture(texture, h_output);

    while (window.isOpen()) {
        handleEvents(window, x_min, x_max, y_min, y_max, needsUpdate, lastMousePos, isDragging);

        if (needsUpdate) {
            computeMandelbrot(d_output, WIDTH, HEIGHT, x_min, x_max, y_min, y_max);
            err = cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
                break;
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
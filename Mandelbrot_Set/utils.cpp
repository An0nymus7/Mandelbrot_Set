#include "utils.h"
#include <iostream>

void initWindow(sf::RenderWindow& window, sf::Texture& texture, sf::Sprite& sprite) {
    sf::ContextSettings settings;
    settings.majorVersion = 3;
    settings.minorVersion = 3;
    window.create(sf::VideoMode(WIDTH, HEIGHT), "Mandelbrot CUDA", sf::Style::Default, settings);
    if (!texture.create(WIDTH, HEIGHT)) {
        std::cerr << "Failed to create texture" << std::endl;
        return;
    }
    sprite.setTexture(texture);
}

void updateTexture(sf::Texture& texture, unsigned char* h_output) {
    texture.update(h_output, WIDTH, HEIGHT, 0, 0);
}

void handleEvents(sf::RenderWindow& window, float& x_min, float& x_max, float& y_min, float& y_max, bool& needsUpdate, sf::Vector2i& lastMousePos, bool& isDragging) {
    sf::Event event;

    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window.close();
        }

        // Zoom with mouse wheel
        if (event.type == sf::Event::MouseWheelScrolled) {
            float zoom_factor = (event.mouseWheelScroll.delta > 0) ? 0.8f : 1.25f; // Zoom in: 0.8, out: 1.25
            sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
            float mx = x_min + (x_max - x_min) * mouse_pos.x / WIDTH;
            float my = y_min + (y_max - y_min) * mouse_pos.y / HEIGHT;

            float new_width = (x_max - x_min) * zoom_factor;
            float new_height = (y_max - y_min) * zoom_factor;
            float new_x_min = mx - new_width * ((float)mouse_pos.x / WIDTH);
            float new_x_max = mx + new_width * (1.0f - (float)mouse_pos.x / WIDTH);
            float new_y_min = my - new_height * ((float)mouse_pos.y / HEIGHT);
            float new_y_max = my + new_height * (1.0f - (float)mouse_pos.y / HEIGHT);

            x_min = new_x_min;
            x_max = new_x_max;
            y_min = new_y_min;
            y_max = new_y_max;
            needsUpdate = true;
        }

        // Panning with mouse drag
        if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
            isDragging = true;
            lastMousePos = sf::Mouse::getPosition(window);
        }
        if (event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Left) {
            isDragging = false;
            needsUpdate = true; // Update after drag ends
        }
        if (event.type == sf::Event::MouseMoved && isDragging) {
            sf::Vector2i currentMousePos = sf::Mouse::getPosition(window);
            float dx = (x_max - x_min) * (lastMousePos.x - currentMousePos.x) / WIDTH;
            float dy = (y_max - y_min) * (lastMousePos.y - currentMousePos.y) / HEIGHT;

            x_min += dx;
            x_max += dx;
            y_min += dy;
            y_max += dy;

            lastMousePos = currentMousePos;
            needsUpdate = true; // Update during drag
        }
    }
}
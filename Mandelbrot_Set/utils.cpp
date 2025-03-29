#include "utils.h"
#include <iostream>

void initWindow(sf::RenderWindow& window, sf::Texture& texture, sf::Sprite& sprite)
{
	sf::ContextSettings settings;
	settings.majorVersion = 3;
	settings.minorVersion = 3;
	window.create(sf::VideoMode(WIDTH, HEIGHT), "Mandelbrot CUDA",sf::Style::Default,settings);
	if (!texture.create(WIDTH, HEIGHT))
	{
		std::cerr << "Failed to create texture" << std::endl;
	}
	texture.create(WIDTH, HEIGHT);
	sprite.setTexture(texture);
}

void updateTexture(sf::Texture& texture, unsigned char* h_output)
{
	texture.update(h_output,WIDTH,HEIGHT,0,0);
}

void handleEvents(sf::RenderWindow& window, float& x_min, float& x_max, float& y_min, float& y_max, bool& needsUpdate)
{
	sf::Event event;
	while (window.pollEvent(event)) {
		if (event.type == sf::Event::Closed)
			window.close();
		if (event.type == sf::Event::MouseWheelScrolled) {
			float zoom_factor = (event.mouseWheelScroll.delta > 0) ? 0.8f : 1.25f;
			sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
			float mx = x_min + (x_max - x_min) * mouse_pos.x / WIDTH;
			float my = y_min + (y_max - y_min) * mouse_pos.y / HEIGHT;
		
			float new_width = (x_max - x_min) * zoom_factor;
			float new_height = (y_max - y_min) * zoom_factor;

			x_min = mx - new_width * ((float)mouse_pos.x / WIDTH);
			x_max = mx + new_width * (1.0f - (float)mouse_pos.x / WIDTH);
			y_min = my - new_height * ((float)mouse_pos.y / HEIGHT);
			y_max = my+ new_height * (1.0f -(float)mouse_pos.y / HEIGHT);

			needsUpdate = true;
		}
	}
}

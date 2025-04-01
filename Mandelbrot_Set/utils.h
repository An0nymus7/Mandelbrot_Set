#include <SFML/Graphics.hpp>

#define WIDTH 1280
#define HEIGHT 720


void initWindow(sf::RenderWindow& window, sf::Texture& texture, sf::Sprite& sprite);
void updateTexture(sf::Texture& texture, unsigned char* h_output);
void handleEvents(sf::RenderWindow& window, float& x_min, float& x_max, float& y_min, float& y_max, bool& needsUpdate, sf::Vector2i& lastMousePos, bool& isDragging);

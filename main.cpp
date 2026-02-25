#include <SFML/Graphics.hpp>
#include <iostream>
#include <string>

int charX {400};
int charY {300};

void writeText(sf::RenderWindow& window, std::string str)
{
	sf::Font font;
	if (!font.loadFromFile("RobotoSlab-Bold.otf")){}
	sf::Text text;
	text.setFont(font);
	text.setString(str);
	text.setCharacterSize(24);
	text.setFillColor(sf::Color::Red);
	window.draw(text);
	return;
}

int main()
{
	sf::RenderWindow window(sf::VideoMode(800, 600), "Malunga");
	
	// Carregando textura
	sf::Texture texture;
	if (!texture.loadFromFile("red.png")){}
	
	// Setando a textura no sprite
	sf::Sprite sprite;
	sprite.setPosition(sf::Vector2f(400.f, 300.f));
	sprite.setTexture(texture);
	
	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			switch (event.type)
			{
				case sf::Event::Closed:
					window.close();
					break;
					
				case sf::Event::KeyPressed:
					switch (event.key.code)
					{
						case sf::Keyboard::A:
							if (charX - 20 < 0);
							else
							{
								sprite.move(-20.f, 0.f);
								charX -= 20;
							}
							break;
							
						case sf::Keyboard::D:
							if (charX + 20 >= 800);
							else
							{
								sprite.move(20.f, 0.f);
								charX += 20;
							}
							break;
							
						case sf::Keyboard::W:
							if (charY - 20 < 0);
							else
							{
								sprite.move(0.f, -20.f);
								charY -= 20;
							}
							break;
							
						case sf::Keyboard::S:
							if (charY + 20 >= 600);
							else
							{
								sprite.move(0.f, 20.f);
								charY += 20;
							}
							break;
							
						default:
							break;
					}
					break;
					
				default:
					break;
			}
		}
		
		window.clear(sf::Color::Black);
		std::string posicao (std::to_string(charX) + " " + std::to_string(charY));
		writeText(window, posicao);
		window.draw(sprite);
		window.display();
	}
	
	return 0;
}

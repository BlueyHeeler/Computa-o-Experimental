#include <SFML/Graphics.hpp>
#include <iostream>
#include <string>
#include <tuple>

int charX {400};
int charY {300};
char lastPressed {'z'};
sf::Time elapsed1;
sf::Time elapsed2;

void writeText(sf::RenderWindow &window, std::string str, std::tuple <float, float> t)
{
	sf::Font font;
	if (!font.loadFromFile("RobotoSlab-Bold.otf")){}
	sf::Text text;
	text.setFont(font);
	text.setString(str);
	text.setCharacterSize(24);
	text.setFillColor(sf::Color::Red);
	text.setOrigin(sf::Vector2f(std::get<0>(t), std::get<1>(t)));
	window.draw(text);
	return;
}

void charMovement(sf::Event &event, sf::Sprite &sprite)
{
	switch (event.key.code)
		{
		case sf::Keyboard::A:
			if (charX - 20 < 0 || lastPressed == 'a');
			else
			{
				sprite.move(-20.f, 0.f);
				charX -= 20;
				lastPressed = 'a';
			}
			break;
							
		case sf::Keyboard::D:
			if (charX + 20 >= 800 || lastPressed == 'd');
			else
			{
				sprite.move(20.f, 0.f);
				charX += 20;
				lastPressed = 'd';
			}
			break;
							
		case sf::Keyboard::W:
			if (charY - 20 < 0 || lastPressed == 'w');
			else
			{
				sprite.move(0.f, -20.f);
				charY -= 20;
				lastPressed = 'w';
			}
			break;
							
		case sf::Keyboard::S:
			if (charY + 20 >= 600 || lastPressed == 's');
			else
			{
				sprite.move(0.f, 20.f);
				charY += 20;
				lastPressed = 's';
			}
			break;
							
		default:
			break;
		}
}

void snakeMovement(sf::Event &event, sf::Sprite &sprite)
{
	switch (lastPressed)
	{
		case 'w':
			if (charY - 20 < 0);
			else
			{
				sprite.move(0.f, -20.f);
				charY -= 20;
				lastPressed = 'w';
			}
			break;
					
		case 's':
			if (charY + 20 >= 600);
			else
			{
				sprite.move(0.f, 20.f);
				charY += 20;
				lastPressed = 's';
			}
			break;
					
		case 'a':
			if (charX - 20 < 0);
			else
			{
				sprite.move(-20.f, 0.f);
				charX -= 20;
				lastPressed = 'a';
			}
			break;
					
		case 'd':
			if (charX + 20 >= 800);
			else
			{
				sprite.move(20.f, 0.f);
				charX += 20;
				lastPressed = 'd';
			}
			break;
		default:
			break;
	}
}

int main()
{
	sf::RenderWindow window(sf::VideoMode(800, 600), "Malunga");
	
	// Carregando textura
	sf::Texture texture;
	if (!texture.loadFromFile("red.png")){}
	
	// Setando a textura no sprite e sua posição
	sf::Sprite sprite;
	sprite.setPosition(sf::Vector2f(400.f, 300.f));
	sprite.setTexture(texture);
	
	// Iniciando o relógio do jogo
	sf::Clock clock;
	
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
					charMovement(event, sprite);
					break;
					
				default:
					break;
			}
		}
		if(clock.getElapsedTime().asSeconds() >= 0.25f)
		{
			snakeMovement(event, sprite);
			clock.restart();
		}
		window.clear(sf::Color::Black);
		//Debug
		std::string posicao (std::to_string(charX) + " " + std::to_string(charY));
		writeText(window, posicao, {0.f, 0.f});
		std::string ultimoPressionado = "";
		ultimoPressionado += lastPressed;
		writeText(window, ultimoPressionado, {-100.f, 0.f});
		window.draw(sprite);
		window.display();
	}
	
	return 0;
}

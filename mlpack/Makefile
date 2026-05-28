all:
	g++ random_forest.cpp -o random_forest -larmadillo -std=c++17
	g++ testar_modelo.cpp -o testar_modelo -larmadillo -std=c++17

feature:
	./random_forest 2
	./testar_modelo 2

sem_feature:
	./random_forest 1
	./testar_modelo 1

media:
	g++ importancia_media_rf.cpp -o importancia_media -larmadillo -std=c++17

media_sem_feature:
	./importancia_media 1

media_feature:
	./importancia_media 2

media_podado_1:
	./importancia_media 3

media_podado_2:
	./importancia_media 4

sintetico:
	python3 ../gerar_sintetico.py

media_augmented:
	./importancia_media 5

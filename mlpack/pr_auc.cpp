#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

using namespace mlpack;
using namespace std;

int main(int argc, char*argv[])
{
	mlpack::RandomSeed(0);
	
	arma::mat train_dataset;
	arma::Row<size_t> train_labels;

	// Acesso é por coluna e linha, ou seja, train_dataset(coluna, linha)
	data::Load("dados_podados_2/treinamento_podado_2.csv", train_dataset, true);
	data::Load("dados_podados_2/labels_treinamento_podado_2.csv", train_labels, true);

	double weightNoRain = 1.0;
	double weightRain = 77.9244 / 22.0756;
	
	arma::rowvec weights(train_labels.n_elem);
	
	for (size_t i = 0; i < train_labels.n_elem; ++i)
	{
		if (train_labels[i] == 0)
			weights[i] = weightNoRain;
		else
			weights[i] = weightRain;
	}
	
	const size_t numClasses = 2;
	const size_t numTrees = 100;
	const size_t minimumLeafSize = 14;
	
	//RandomForest<> rf;
	//rf.Train(train_dataset, train_labels, numClasses, weights, numTrees, minimumLeafSize);
	
	
}

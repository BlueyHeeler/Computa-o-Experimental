#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

using namespace mlpack;
using namespace std;

int main(int argc, char *argv[]) {
    int tipo = 0;
	if (argc > 1)
    {
        tipo = stoi(argv[1]);
    }
    else    {
        cout << "Uso: " << argv[0] << " tipo 1 ou 2" << endl;
        return 1;
    }
	random_device rd; 
    mt19937 gen(rd()); 
    uniform_int_distribution<> distr(1, 10000); 
    mlpack::RandomSeed(distr(gen));
    
    // 1. Inicialização das matrizes
    arma::mat dataset;
    arma::Row<size_t> labels; // 0 = No Rain, 1 = Rain

    // Carregamento dos dados (lembre-se: dataset transposto)
    if(tipo == 1)
    {
        data::Load("dados_sem_feature/treinamento_tratado_sem.csv", dataset, true);
        data::Load("dados_sem_feature/labels_treinamento_sem.csv", labels, true);
    }else if(tipo == 2)
    {
        data::Load("dados_com_feature/treinamento_tratado_com.csv", dataset, true);
        data::Load("dados_com_feature/labels_treinamento_com.csv", labels, true);
    }else {
        cout << "Tipo inválido. Use 1 ou 2." << endl;
        return 1;
    }

    // 2. Cálculo e Atribuição dos Pesos
    double weightNoRain = 1.0;
    double weightRain = 77.9244 / 22.0756; // Aproximadamente 3.53

    // Cria um vetor de pesos com o exato mesmo tamanho do vetor de labels
    arma::rowvec weights(labels.n_elem);

    // Itera sobre as labels para aplicar o peso correspondente a cada linha
    for (size_t i = 0; i < labels.n_elem; ++i) {
        if (labels[i] == 0) {
            weights[i] = weightNoRain;
        } else {
            weights[i] = weightRain;
        }
    }

    // 3. Configuração dos Hiperparâmetros
    const size_t numClasses = 2; // Sim e Não
    const size_t numTrees = 100;  // Ajuste conforme necessário
    const size_t minimumLeafSize = 14;

    // 4. Treinamento com Pesos
    RandomForest<> rf;
    
    cout << "Iniciando o treinamento balanceado..." << endl;
    
    // A sobrecarga do método Train que aceita o vetor 'weights'
    rf.Train(dataset, labels, numClasses, weights, numTrees, minimumLeafSize);
    
    cout << "Treinamento concluído com sucesso!" << endl;

    // (Opcional) Salvar o modelo
    data::Save("modelo_clima_rf.bin", "modelo", rf);

    return 0;
}

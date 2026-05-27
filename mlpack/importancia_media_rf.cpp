#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <algorithm>
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

using namespace mlpack;
using namespace std;

// Calcula o F1-Score para um dado threshold
double calcularF1Score(mlpack::RandomForest<>& rf,
                       const arma::mat& dados,
                       const arma::Row<size_t>& labels_reais,
                       double threshold)
{
    arma::Row<size_t> previsoes_padrao;
    arma::mat probabilities;
    rf.Classify(dados, previsoes_padrao, probabilities);

    size_t truePositives  = 0;
    size_t falsePositives = 0;
    size_t falseNegatives = 0;

    for (size_t i = 0; i < labels_reais.n_elem; ++i)
    {
        size_t previsao = (probabilities(1, i) >= threshold) ? 1 : 0;
        if      (previsao == 1 && labels_reais[i] == 1) truePositives++;
        else if (previsao == 1 && labels_reais[i] == 0) falsePositives++;
        else if (previsao == 0 && labels_reais[i] == 1) falseNegatives++;
    }

    double recall   = (truePositives + falseNegatives > 0)
                      ? (double)truePositives / (truePositives + falseNegatives) : 0.0;
    double precisao = (truePositives + falsePositives > 0)
                      ? (double)truePositives / (truePositives + falsePositives) : 0.0;

    if (recall + precisao == 0.0) return 0.0;
    return 2.0 * (recall * precisao) / (recall + precisao);
}

int main(int argc, char* argv[])
{
    // -----------------------------------------------------------------------
    // 0. Argumentos
    // -----------------------------------------------------------------------
    int tipo = 0;
    if (argc > 1)
        tipo = stoi(argv[1]);
    else {
        cout << "Uso: " << argv[0] << " <tipo>  (1 = sem feature eng., 2 = com feature eng., 3 = dados podados 1., 4 = dados podados 2.)" << endl;
        return 1;
    }
    if (tipo != 1 && tipo != 2 && tipo != 3 && tipo != 4) {
        cout << "Tipo inválido. Use 1 ou 2 ou 3." << endl;
        return 1;
    }

    // -----------------------------------------------------------------------
    // 1. Caminhos de dados
    // -----------------------------------------------------------------------
    string path_train_data, path_train_labels;
    string path_val_data,   path_val_labels;

    if (tipo == 1) {
        path_train_data   = "dados_sem_feature/treinamento_tratado_sem.csv";
        path_train_labels = "dados_sem_feature/labels_treinamento_sem.csv";
        path_val_data     = "dados_sem_feature/validacao_tratado_sem.csv";
        path_val_labels   = "dados_sem_feature/labels_validacao_sem.csv";
    } else if(tipo == 2) {
        path_train_data   = "dados_com_feature/treinamento_tratado_com.csv";
        path_train_labels = "dados_com_feature/labels_treinamento_com.csv";
        path_val_data     = "dados_com_feature/validacao_tratado_com.csv";
        path_val_labels   = "dados_com_feature/labels_validacao_com.csv";
    } else if(tipo == 3){
    	path_train_data   = "dados_podados/treinamento_podado.csv";
        path_train_labels = "dados_podados/labels_treinamento_podado.csv";
        path_val_data     = "dados_podados/validacao_podado.csv";
        path_val_labels   = "dados_podados/labels_validacao_podado.csv";
    } else if(tipo == 4){
    	path_train_data   = "dados_podados_2/treinamento_podado_2.csv";
        path_train_labels = "dados_podados_2/labels_treinamento_podado_2.csv";
        path_val_data     = "dados_podados_2/validacao_podado_2.csv";
        path_val_labels   = "dados_podados_2/labels_validacao_podado_2.csv";
    }

    // -----------------------------------------------------------------------
    // 2. Carregamento dos dados (feito apenas uma vez)
    // -----------------------------------------------------------------------
    arma::mat trainData, valData;
    arma::Row<size_t> trainLabels, valLabels;

    cout << "Carregando dados de treinamento..." << endl;
    data::Load(path_train_data,   trainData,   true);
    data::Load(path_train_labels, trainLabels, true);

    cout << "Carregando dados de validação..." << endl;
    data::Load(path_val_data,   valData,   true);
    data::Load(path_val_labels, valLabels, true);

    cout << "Features no treinamento: " << trainData.n_rows << endl;
    cout << "Amostras de treinamento: " << trainData.n_cols << endl;
    cout << "Features na validação:   " << valData.n_rows   << endl;
    cout << "Amostras de validação:   " << valData.n_cols   << endl;

    // -----------------------------------------------------------------------
    // 3. Nomes das features (Tentar pegar pelo header do arquivo ao invés de manual)
    // -----------------------------------------------------------------------
    vector<string> feature_names;
    if (tipo == 1) {
        feature_names = {
            "Date_x", "Date_y",
            "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
            "WindGustDir_x", "WindGustDir_y", "WindGustSpeed",
            "WindDir9am_x", "WindDir9am_y",
            "WindDir3pm_x", "WindDir3pm_y",
            "WindSpeed9am", "WindSpeed3pm",
            "Humidity9am", "Humidity3pm",
            "Pressure9am", "Pressure3pm",
            "Cloud9am", "Cloud3pm",
            "Temp9am", "Temp3pm",
            "RainToday"
        }; // 25 features
    } else if (tipo == 2){
        feature_names = {
            "Date_x", "Date_y",
            "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
            "WindGustDir_x", "WindGustDir_y", "WindGustSpeed",
            "WindDir9am_x", "WindDir9am_y",
            "WindDir3pm_x", "WindDir3pm_y",
            "WindSpeed9am", "WindSpeed3pm",
            "Humidity9am", "Humidity3pm",
            "Pressure9am", "Pressure3pm",
            "Cloud9am", "Cloud3pm",
            "Temp9am", "Temp3pm",
            "DewDiff3pm", "DewDiff9am",
            "TempRise", "PressureDrop",
            "RainToday"
        	}; // 29 features
    } else if (tipo == 3){
    	feature_names = {
            "Date_x", "Date_y",
            "Rainfall", "Evaporation", "Sunshine",
            "WindGustDir_x", "WindGustDir_y", "WindGustSpeed",
            "WindDir9am_x", "WindDir9am_y",
            "WindDir3pm_x", "WindDir3pm_y",
            "WindSpeed9am", "WindSpeed3pm",
            "Humidity9am", "Humidity3pm",
            "Pressure9am", "Pressure3pm",
            "Cloud3pm",
            "Temp9am",
            "DewDiff3pm", "DewDiff9am",
            "TempRise", "PressureDrop",
            "RainToday"	    	
        }; // 23 features
    } else if (tipo == 4){
    	feature_names = {
            "Date_x", "Date_y",
            "Rainfall", "Sunshine",
            "WindGustDir_x", "WindGustSpeed",
            "WindDir9am_x", "WindDir9am_y",
            "WindDir3pm_x", "WindDir3pm_y",
            "WindSpeed3pm",
            "Humidity9am", "Humidity3pm",
            "Pressure9am", "Pressure3pm",
            "Cloud3pm",
            "DewDiff3pm", "DewDiff9am",
            "TempRise", "PressureDrop"  	
        }; // 21 features   
    }

    const size_t numFeatures = valData.n_rows;

    // -----------------------------------------------------------------------
    // 4. Pesos de classe (mesmo cálculo do random_forest.cpp original)
    // -----------------------------------------------------------------------
    const double weightNoRain = 1.0;
    const double weightRain   = 77.9244 / 22.0756; // ~3.53

    arma::rowvec weights(trainLabels.n_elem);
    for (size_t i = 0; i < trainLabels.n_elem; ++i)
        weights[i] = (trainLabels[i] == 0) ? weightNoRain : weightRain;

    // -----------------------------------------------------------------------
    // 5. Hiperparâmetros
    // -----------------------------------------------------------------------
    const size_t numClasses       = 2;
    const size_t numTrees         = 100;
    const size_t minimumLeafSize  = 14;
    const double threshold        = 0.30;
    const int    NUM_RUNS         = 1000;

    // -----------------------------------------------------------------------
    // 6. Acumuladores
    // -----------------------------------------------------------------------
    vector<double> importancia_acumulada(numFeatures, 0.0);
    vector<double> f1_runs;
    f1_runs.reserve(NUM_RUNS);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distr(1, 1000000);

    // -----------------------------------------------------------------------
    // 7. Loop principal: 100 execuções
    // -----------------------------------------------------------------------
    for (int run = 1; run <= NUM_RUNS; ++run)
    {
        cout << "\n[Run " << run << "/" << NUM_RUNS << "] Treinando..." << flush;

        // Semente aleatória diferente a cada rodada
        mlpack::RandomSeed(distr(gen));

        RandomForest<> rf;
        rf.Train(trainData, trainLabels, numClasses, weights, numTrees, minimumLeafSize);

        // F1-Score base desta rodada
        double f1_base = calcularF1Score(rf, valData, valLabels, threshold);
        f1_runs.push_back(f1_base);

        cout << " F1-base = " << f1_base * 100.0 << "%" << flush;

        // Importância por permutação para cada feature
        for (size_t feat = 0; feat < numFeatures; ++feat)
        {
            arma::mat dados_baguncados = valData;
            dados_baguncados.row(feat) = arma::shuffle(dados_baguncados.row(feat));

            double f1_baguncado = calcularF1Score(rf, dados_baguncados, valLabels, threshold);
            importancia_acumulada[feat] += (f1_base - f1_baguncado);
        }
    }

    // -----------------------------------------------------------------------
    // 8. Médias finais
    // -----------------------------------------------------------------------
    double f1_media = accumulate(f1_runs.begin(), f1_runs.end(), 0.0) / NUM_RUNS;
    double f1_min   = *min_element(f1_runs.begin(), f1_runs.end());
    double f1_max   = *max_element(f1_runs.begin(), f1_runs.end());

    // Calcula variância do F1
    double variancia = 0.0;
    for (double v : f1_runs) variancia += (v - f1_media) * (v - f1_media);
    variancia /= NUM_RUNS;
    double desvio_padrao = sqrt(variancia);

    vector<double> importancia_media(numFeatures);
    for (size_t i = 0; i < numFeatures; ++i)
        importancia_media[i] = importancia_acumulada[i] / NUM_RUNS;

    // Ordena por importância (decrescente) para exibição
    vector<size_t> indices(numFeatures);
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&](size_t a, size_t b){
        return importancia_media[a] > importancia_media[b];
    });

    // -----------------------------------------------------------------------
    // 9. Exibição dos resultados
    // -----------------------------------------------------------------------
    cout << "\n\n============================================================" << endl;
    cout << "   RESULTADOS FINAIS — MÉDIA DE " << NUM_RUNS << " RODADAS" << endl;
    cout << "   Tipo: " << (tipo == 1 ? "SEM feature engineering" : "COM feature engineering") << endl;
    cout << "   Threshold: " << threshold * 100.0 << "%" << endl;
    cout << "============================================================" << endl;

    cout << "\n--- Estatísticas do F1-Score (chuva) ---" << endl;
    cout << "  Média:         " << f1_media * 100.0    << "%" << endl;
    cout << "  Desvio Padrão: " << desvio_padrao * 100.0 << "%" << endl;
    cout << "  Mínimo:        " << f1_min * 100.0       << "%" << endl;
    cout << "  Máximo:        " << f1_max * 100.0       << "%" << endl;

    cout << "\n--- Importância Média das Features (por permutação) ---" << endl;
    cout << "  (ordenado da mais para a menos importante)" << endl;
    cout << "  Queda no F1 = quanto o modelo piora ao remover a feature" << endl;
    cout << endl;

    for (size_t rank = 0; rank < numFeatures; ++rank)
    {
        size_t idx = indices[rank];
        // Padding manual para alinhar colunas
        string nome = feature_names[idx];
        nome.resize(20, ' ');
        cout << "  " << (rank + 1) << ".\t" << nome
             << "\tQueda média no F1: "
             << importancia_media[idx] * 100.0 << "%" << endl;
    }

    cout << "\n============================================================\n" << endl;

    return 0;
}

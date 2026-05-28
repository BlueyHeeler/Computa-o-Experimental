#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

using namespace mlpack;
using namespace std;
// Adicione ANTES do main()
double calcularF1Score(mlpack::RandomForest<>& rf, const arma::mat& dados, const arma::Row<size_t>& labels_reais, double threshold) {
    arma::Row<size_t> previsoes_padrao;
    arma::mat probabilities;
    
    // Roda as previsões nos dados fornecidos
    rf.Classify(dados, previsoes_padrao, probabilities);

    size_t truePositives = 0;
    size_t falsePositives = 0;
    size_t falseNegatives = 0;

    for (size_t i = 0; i < labels_reais.n_elem; ++i) {
        size_t previsao = (probabilities(1, i) >= threshold) ? 1 : 0;
        
        if (previsao == 1 && labels_reais[i] == 1) truePositives++;
        else if (previsao == 1 && labels_reais[i] == 0) falsePositives++;
        else if (previsao == 0 && labels_reais[i] == 1) falseNegatives++;
    }

    double recall = (truePositives + falseNegatives > 0) ? (double)truePositives / (truePositives + falseNegatives) : 0;
    double precisao = (truePositives + falsePositives > 0) ? (double)truePositives / (truePositives + falsePositives) : 0;
    
    if (recall + precisao == 0) return 0.0;
    
    return 2.0 * (recall * precisao) / (recall + precisao);
}

int main(int argc, char* argv[]) {
    int tipo = 0;
    if (argc > 1) {
        tipo = stoi(argv[1]);
    } else {
        cout << "Uso: " << argv[0] << " tipo (1: sem feature, 2: com feature)" << endl;
        return 1;
    }

    // 1. Carregar o modelo treinado
    RandomForest<> rf;
    cout << "Carregando o modelo..." << endl;
    data::Load("modelo_clima_rf.bin", "modelo", rf);

    // 2. Carregar os dados de validação
    arma::mat valData;
    arma::Row<size_t> valLabels;
    
    if (tipo == 1) {
        cout << "Carregando dados de validação SEM feature engineering..." << endl;
        data::Load("dados_sem_feature/validacao_tratado_sem.csv", valData, true);
        cout << valData.n_rows << " features\n";
        data::Load("dados_sem_feature/labels_validacao_sem.csv", valLabels, true);
    } else if (tipo == 2) {
        cout << "Carregando dados de validação COM feature engineering..." << endl;
        data::Load("dados_com_feature/validacao_tratado_com.csv", valData, true);
        cout << valData.n_rows << " features\n";
        data::Load("dados_com_feature/labels_validacao_com.csv", valLabels, true);
    } else {
        cout << "Tipo inválido. Use 1 ou 2." << endl;
        return 1;
    }

    // 3. Matrizes para guardar a resposta padrão e as probabilidades
    arma::Row<size_t> previsoes_padrao;
    arma::mat probabilities; // Aqui a mágica acontece

    cout << "Realizando previsões detalhadas..." << endl;
    rf.Classify(valData, previsoes_padrao, probabilities);

    // 4. Aplicando o nosso Threshold (Limiar) Customizado
    arma::Row<size_t> minhas_previsoes(valLabels.n_elem);
    
    // DEFINA O LIMIAR AQUI: 0.30 significa que se tiver 30% de chance, já soamos o alarme!
    double threshold = 0.30; 

    for (size_t i = 0; i < valLabels.n_elem; ++i) {
        // A linha 1 da matriz guarda a probabilidade da classe 1 (Chuva)
        double chance_de_chuva = probabilities(1, i);
        //cout << probabilities(0, i) << "\n";

        if (chance_de_chuva >= threshold) {
        	//cout << chance_de_chuva << "--- CHUVA !!!\n";
            minhas_previsoes[i] = 1; // Forçamos a previsão para Chuva
        } else {
        	//cout << chance_de_chuva << "--- SOL !!!\n";
            minhas_previsoes[i] = 0; // Sol
        }
    }

    // 5. Construir a Matriz de Confusão usando nossas previsões
    size_t truePositives = 0;  
    size_t falsePositives = 0; 
    size_t trueNegatives = 0;  
    size_t falseNegatives = 0; 

    for (size_t i = 0; i < valLabels.n_elem; ++i) {
        if (minhas_previsoes[i] == 1 && valLabels[i] == 1) {
            truePositives++;
        } else if (minhas_previsoes[i] == 1 && valLabels[i] == 0) {
            falsePositives++;
        } else if (minhas_previsoes[i] == 0 && valLabels[i] == 0) {
            trueNegatives++;
        } else if (minhas_previsoes[i] == 0 && valLabels[i] == 1) {
            falseNegatives++;
        }
    }

    // 6. Exibir Resultados
    double acuraciaTotal = (double)(truePositives + trueNegatives) / valLabels.n_elem * 100.0;
    
    // Evita divisão por zero no cálculo de Recall e Precisão
    double recall = (truePositives + falseNegatives > 0) ? 
                    (double)truePositives / (truePositives + falseNegatives) * 100.0 : 0;
    double precisao = (truePositives + falsePositives > 0) ? 
                      (double)truePositives / (truePositives + falsePositives) * 100.0 : 0;

    cout << "\n=============================================" << endl;
    cout << "         RESULTADOS (THRESHOLD: " << threshold * 100 << "%)" << endl;
    cout << "=============================================" << endl;
    cout << "Total de dias avaliados: " << valLabels.n_elem << endl;
    cout << "Acurácia Global: " << acuraciaTotal << "%" << endl;
    cout << "Recall (Sensibilidade Chuva): " << recall << "%" << endl;
    cout << "Precisão (Acerto do Alarme): " << precisao << "%" << endl;
    cout << "F1-Score: " << (recall + precisao > 0 ? 2 * (recall * precisao) / (recall + precisao) : 0) << "%" << endl;
    
    cout << "\n--- Matriz de Confusão ---" << endl;
    cout << "                 | Realidade: Choveu | Realidade: Sol" << endl;
    cout << "-----------------|-------------------|----------------" << endl;
    cout << "Previu: Chuva(1) | TP: " << truePositives << "          | FP: " << falsePositives << " (Alarme Falso)" << endl;
    cout << "Previu: Sol (0)  | FN: " << falseNegatives << " (Não avisou)| TN: " << trueNegatives << endl;
    cout << "=============================================\n" << endl;
	
	// Adicione no FINAL do main(), logo antes de "return 0;"
    
    cout << "\n=============================================" << endl;
    cout << "     IMPORTÂNCIA DAS VARIÁVEIS (PERMUTAÇÃO)" << endl;
    cout << "=============================================" << endl;

    // 1. Calcula o F1-Score base (com todos os dados intactos)
    double f1_base = calcularF1Score(rf, valData, valLabels, threshold);

    // Vetor com os nomes das features NA ORDEM CORRETA
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
} else {
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
}


    // 2. Loop pelas features (linhas da matriz no mlpack/armadillo)
    for (size_t feature_idx = 0; feature_idx < (valData.n_rows); ++feature_idx) {
        
        // Copia os dados para não estragar a matriz original
        arma::mat dados_baguncados = valData;
        
        // Embaralha (shuffle) apenas os valores desta feature específica
        dados_baguncados.row(feature_idx) = arma::shuffle(dados_baguncados.row(feature_idx));
        
        // Calcula o F1-Score com a feature inutilizada
        double f1_baguncado = calcularF1Score(rf, dados_baguncados, valLabels, threshold);
        
        // A importância é o tamanho do estrago. Se for alto, a feature é vital.
        double queda_f1 = f1_base - f1_baguncado;
        
        cout << "Feature " << feature_names[feature_idx] << " | Queda no F1: " << queda_f1 * 100 << "%" << endl;
    }
    
    return 0;
}

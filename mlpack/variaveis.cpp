#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

using namespace mlpack;
using namespace std;

int main(int argc, char* argv[])
{
    mlpack::RandomSeed(0);

    RandomForest<> rf;

    arma::mat train_dataset;
    arma::Row<size_t> train_labels;

    arma::mat validation_dataset;
    arma::Row<size_t> validation_labels;

    arma::Row<size_t> predictions;
    arma::mat probabilities;

    data::Load("dados_podados_2/treinamento_podado_2.csv",          train_dataset,      true);
    data::Load("dados_podados_2/labels_treinamento_podado_2.csv",   train_labels,       true);

    data::Load("dados_podados_2/validacao_podado_2.csv",            validation_dataset, true);
    data::Load("dados_podados_2/labels_validacao_podado_2.csv",     validation_labels,  true); // ✅ corrigido

    double weightNoRain = 1.0;
    double weightRain   = 77.9244 / 22.0756;

    arma::rowvec weights(train_labels.n_elem);
    for (size_t i = 0; i < train_labels.n_elem; ++i)
        weights[i] = (train_labels[i] == 0) ? weightNoRain : weightRain;

    const size_t numClasses       = 2;
    const size_t numTrees         = 100;
    const size_t minimumLeafSize  = 14;

    rf.Train(train_dataset, train_labels, numClasses, weights, numTrees, minimumLeafSize);
    rf.Classify(validation_dataset, predictions, probabilities);

    arma::Row<size_t> my_predictions(validation_labels.n_elem);
    double threshold = 0.30;

	ofstream file("probabilidades.csv");
	
    for (size_t i = 0; i < validation_labels.n_elem; ++i)
    {
        double chance_de_chuva = probabilities(1, i);
        file << probabilities(0, i) << " " << probabilities(1, i) << "\n";
        my_predictions[i] = (chance_de_chuva >= threshold) ? 1 : 0;
    }

    size_t truePositives  = 0;
    size_t falsePositives = 0;
    size_t trueNegatives  = 0;
    size_t falseNegatives = 0;

    for (size_t i = 0; i < validation_labels.n_elem; ++i)
    {
        if      (my_predictions[i] == 1 && validation_labels[i] == 1) truePositives++;
        else if (my_predictions[i] == 1 && validation_labels[i] == 0) falsePositives++;
        else if (my_predictions[i] == 0 && validation_labels[i] == 0) trueNegatives++;
        else if (my_predictions[i] == 0 && validation_labels[i] == 1) falseNegatives++;
    }

    double acuraciaTotal = (double)(truePositives + trueNegatives) / validation_labels.n_elem * 100.0;
    double recall   = (truePositives + falseNegatives > 0)
                      ? (double)truePositives / (truePositives + falseNegatives) * 100.0 : 0;
    double precisao = (truePositives + falsePositives > 0)
                      ? (double)truePositives / (truePositives + falsePositives) * 100.0 : 0;
    double f1       = (recall + precisao > 0)
                      ? 2.0 * (recall * precisao) / (recall + precisao) : 0;

    cout << "\n=============================================" << endl;
    cout << "         RESULTADOS (THRESHOLD: " << threshold * 100 << "%)" << endl;
    cout << "=============================================" << endl;
    cout << "Total de dias avaliados:        " << validation_labels.n_elem << endl;
    cout << "Acurácia Global:                " << acuraciaTotal << "%" << endl;
    cout << "Recall  (Sensibilidade Chuva):  " << recall        << "%" << endl;
    cout << "Precisão (Acerto do Alarme):    " << precisao      << "%" << endl;
    cout << "F1-Score:                       " << f1            << "%" << endl;

    cout << "\n--- Matriz de Confusão ---" << endl;
    cout << "                 | Realidade: Choveu | Realidade: Sol" << endl;
    cout << "-----------------|-------------------|----------------" << endl;
    cout << "Previu: Chuva(1) | TP: " << truePositives  << "          | FP: " << falsePositives << " (Alarme Falso)" << endl;
    cout << "Previu: Sol  (0) | FN: " << falseNegatives << " (Não avisou)| TN: " << trueNegatives << endl;
    cout << "=============================================\n" << endl;

    // =========================================================
    //   MÓDULO DE BUSCA EM GRADE (GRID SEARCH)
    //   Testa combinações de numTrees x minimumLeafSize
    //   e retorna a de maior F1-Score
    // =========================================================

    const std::vector<size_t> candidatesTrees    = {50, 100, 150, 200, 300};
    const std::vector<size_t> candidatesLeafSize = {5, 10, 14, 20, 30, 50};

    struct GridResult {
        size_t trees;
        size_t leafSize;
        double f1;
        double recall;
        double precision;
        double accuracy;
    };

    GridResult best = {0, 0, -1.0, 0.0, 0.0, 0.0};
    std::vector<GridResult> allResults;

    const size_t totalCombinations = candidatesTrees.size() * candidatesLeafSize.size();
    size_t currentCombo = 0;

    cout << "==============================================" << endl;
    cout << "   INICIANDO GRID SEARCH (" << totalCombinations << " combinações)" << endl;
    cout << "==============================================" << endl;
    cout << left
         << setw(8)  << "Trees"
         << setw(12) << "LeafSize"
         << setw(10) << "F1(%)"
         << setw(10) << "Recall(%)"
         << setw(14) << "Precisão(%)"
         << setw(12) << "Acurácia(%)"
         << endl;
    cout << string(66, '-') << endl;

    for (size_t nTrees : candidatesTrees)
    {
        for (size_t leafSize : candidatesLeafSize)
        {
            ++currentCombo;

            RandomForest<> rfGrid;
            rfGrid.Train(train_dataset, train_labels, numClasses, weights, nTrees, leafSize);

            arma::Row<size_t> gridPreds;
            arma::mat         gridProbs;
            rfGrid.Classify(validation_dataset, gridPreds, gridProbs);

            // Aplica o mesmo threshold já definido
            arma::Row<size_t> gridMyPreds(validation_labels.n_elem);
            for (size_t i = 0; i < validation_labels.n_elem; ++i)
                gridMyPreds[i] = (gridProbs(1, i) >= threshold) ? 1 : 0;

            size_t tp = 0, fp = 0, tn = 0, fn = 0;
            for (size_t i = 0; i < validation_labels.n_elem; ++i)
            {
                if      (gridMyPreds[i] == 1 && validation_labels[i] == 1) tp++;
                else if (gridMyPreds[i] == 1 && validation_labels[i] == 0) fp++;
                else if (gridMyPreds[i] == 0 && validation_labels[i] == 0) tn++;
                else if (gridMyPreds[i] == 0 && validation_labels[i] == 1) fn++;
            }

            double gAcc  = (double)(tp + tn) / validation_labels.n_elem * 100.0;
            double gRec  = (tp + fn > 0) ? (double)tp / (tp + fn) * 100.0 : 0.0;
            double gPrec = (tp + fp > 0) ? (double)tp / (tp + fp) * 100.0 : 0.0;
            double gF1   = (gRec + gPrec > 0)
                           ? 2.0 * (gRec * gPrec) / (gRec + gPrec) : 0.0;

            GridResult res = {nTrees, leafSize, gF1, gRec, gPrec, gAcc};
            allResults.push_back(res);

            cout << left
                 << setw(8)  << nTrees
                 << setw(12) << leafSize
                 << setw(10) << fixed << setprecision(2) << gF1
                 << setw(10) << gRec
                 << setw(14) << gPrec
                 << setw(12) << gAcc
                 << (res.f1 > best.f1 ? "  ← novo melhor" : "")
                 << endl;

            if (gF1 > best.f1)
                best = res;
        }
    }

    cout << string(66, '=') << endl;
    cout << "\n🏆  MELHOR COMBINAÇÃO ENCONTRADA:" << endl;
    cout << "----------------------------------------------" << endl;
    cout << "  numTrees        = " << best.trees    << endl;
    cout << "  minimumLeafSize = " << best.leafSize << endl;
    cout << "  F1-Score        = " << fixed << setprecision(4) << best.f1        << "%" << endl;
    cout << "  Recall          = " << best.recall    << "%" << endl;
    cout << "  Precisão        = " << best.precision << "%" << endl;
    cout << "  Acurácia Global = " << best.accuracy  << "%" << endl;
    cout << "=============================================\n" << endl;

    return 0;
}

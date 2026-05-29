#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include "csv.h"
#include <ctime>

using namespace std;

pair<double, double> direction_to_pair(string direction) {
    int dir_index = -1;
    if (direction == "N")        dir_index = 0;
    else if (direction == "NNE") dir_index = 1;
    else if (direction == "NE")  dir_index = 2;
    else if (direction == "ENE") dir_index = 3;
    else if (direction == "E")   dir_index = 4;
    else if (direction == "ESE") dir_index = 5;
    else if (direction == "SE")  dir_index = 6;
    else if (direction == "SSE") dir_index = 7;
    else if (direction == "S")   dir_index = 8;
    else if (direction == "SSW") dir_index = 9;
    else if (direction == "SW")  dir_index = 10;
    else if (direction == "WSW") dir_index = 11;
    else if (direction == "W")   dir_index = 12;
    else if (direction == "WNW") dir_index = 13;
    else if (direction == "NW")  dir_index = 14;
    else if (direction == "NNW") dir_index = 15;

    if (dir_index == -1) return {0.0, 0.0};

    double x = cos(2 * M_PI * dir_index / 16);
    double y = sin(2 * M_PI * dir_index / 16);
    return {x, y};
}

double dew_diff(string temperatura, string umidade) {
    const double a = 17.27;
    const double b = 237.7;
    double temperaturaAtual = stod(temperatura);
    double umidadeAtual = stod(umidade);
    double alpha = ((a * temperaturaAtual) / (b + temperaturaAtual)) + log(umidadeAtual / 100.0);
    double pontoOrvalho = (b * alpha) / (a - alpha);
    return temperaturaAtual - pontoOrvalho;
}

int date_to_day_of_year(const string& date) {
    // Formato esperado: YYYY-MM-DD
    int year  = stoi(date.substr(0, 4));
    int month = stoi(date.substr(5, 2));
    int day   = stoi(date.substr(8, 2));

    struct tm t = {};
    t.tm_year = year - 1900;
    t.tm_mon  = month - 1;  // tm_mon: 0-11
    t.tm_mday = day;
    mktime(&t);             // preenche tm_yday automaticamente

    return t.tm_yday;       // retorna 0-364 (já ignora dia 366)
}

// BUG 1 CORRIGIDO: ofstream passado por referência (não é copiável)
void _tratamento_(string arquivo, ofstream& arquivo_saida, ofstream& labels_saida, int tipo)
{
    io::CSVReader<22> in(arquivo);
    in.read_header(io::ignore_extra_column, "Date", "MinTemp", "MaxTemp", "Rainfall", "Evaporation",
                   "Sunshine", "WindGustDir", "WindGustSpeed", "WindDir9am",
                   "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
                   "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am",
                   "Cloud3pm", "Temp9am", "Temp3pm", "RainToday", "RainTomorrow");

    string date, minTemp, maxTemp, rainfall, evaporation, sunshine, windGustDir, windGustSpeed,
           windDir9am, windDir3pm, windSpeed9am, windSpeed3pm, humidity9am, humidity3pm,
           pressure9am, pressure3pm, cloud9am, cloud3pm, temp9am, temp3pm, rainToday, rainTomorrow;

    double data_formatada_x = 0;
    double data_formatada_y = 0;
    int count_year_days = 0;
    int linhas_deletadas = 0;

    while(in.read_row(date, minTemp, maxTemp, rainfall, evaporation, sunshine,
                      windGustDir, windGustSpeed, windDir9am, windDir3pm,
                      windSpeed9am, windSpeed3pm, humidity9am, humidity3pm,
                      pressure9am, pressure3pm, cloud9am, cloud3pm,
                      temp9am, temp3pm, rainToday, rainTomorrow)) {

        if (date == "NA" || minTemp == "NA" || maxTemp == "NA" || rainfall == "NA" ||
            evaporation == "NA" || sunshine == "NA" || windGustDir == "NA" || windGustSpeed == "NA" ||
            windDir9am == "NA" || windDir3pm == "NA" || windSpeed9am == "NA" || windSpeed3pm == "NA" ||
            humidity9am == "NA" || humidity3pm == "NA" || pressure9am == "NA" || pressure3pm == "NA" ||
            cloud9am == "NA" || cloud3pm == "NA" || temp9am == "NA" || temp3pm == "NA" ||
            rainToday == "NA" || rainTomorrow == "NA" || date.substr(5, 5) == "02-29") {
            linhas_deletadas++;

        } else {
            int day_of_year = date_to_day_of_year(date);
			data_formatada_x = cos(2 * M_PI * day_of_year / 365);
			data_formatada_y = sin(2 * M_PI * day_of_year / 365);

            pair<double, double> windGustDir_pair = direction_to_pair(windGustDir);
            pair<double, double> windDir9am_pair  = direction_to_pair(windDir9am);
            pair<double, double> windDir3pm_pair  = direction_to_pair(windDir3pm);

            double diferencaManha = dew_diff(temp9am, humidity9am);
            double diferencaTarde = dew_diff(temp3pm, humidity3pm);
            double diff_temp  = stod(temp3pm) - stod(temp9am);
            double diff_press = stod(pressure3pm) - stod(pressure9am);

            if (tipo == 1)
                arquivo_saida << data_formatada_x << ","
                              << data_formatada_y << ","
                              << minTemp << ","
                              << maxTemp << ","
                              << rainfall << ","
                              << evaporation << ","
                              << sunshine << ","
                              << windGustDir_pair.first << ","
                              << windGustDir_pair.second << ","
                              << windGustSpeed << ","
                              << windDir9am_pair.first << ","
                              << windDir9am_pair.second << ","
                              << windDir3pm_pair.first << ","
                              << windDir3pm_pair.second << ","
                              << windSpeed9am << ","
                              << windSpeed3pm << ","
                              << humidity9am << ","
                              << humidity3pm << ","
                              << pressure9am << ","
                              << pressure3pm << ","
                              << cloud9am << ","
                              << cloud3pm << ","
                              << temp9am << ","
                              << temp3pm << ","
                              << (rainToday == "Yes" ? 1 : 0) << "\n";
            else if (tipo == 2)
                arquivo_saida << data_formatada_x << ","
                              << data_formatada_y << ","
                              << minTemp << ","
                              << maxTemp << ","
                              << rainfall << ","
                              << evaporation << ","
                              << sunshine << ","
                              << windGustDir_pair.first << ","
                              << windGustDir_pair.second << ","
                              << windGustSpeed << ","
                              << windDir9am_pair.first << ","
                              << windDir9am_pair.second << ","
                              << windDir3pm_pair.first << ","
                              << windDir3pm_pair.second << ","
                              << windSpeed9am << ","
                              << windSpeed3pm << ","
                              << humidity9am << ","
                              << humidity3pm << ","
                              << pressure9am << ","
                              << pressure3pm << ","
                              << cloud9am << ","
                              << cloud3pm << ","
                              << temp9am << ","
                              << temp3pm << ","
                              << diferencaTarde << ","
                              << diferencaManha << ","
                              << diff_temp << ","
                              << diff_press << ","
                              << (rainToday == "Yes" ? 1 : 0) << "\n";
            else if (tipo == 3)
                arquivo_saida << data_formatada_x << ","
                              << data_formatada_y << ","
                              << rainfall << ","
                              << evaporation << ","
                              << sunshine << ","
                              << windGustDir_pair.first << ","
                              << windGustDir_pair.second << ","
                              << windGustSpeed << ","
                              << windDir9am_pair.first << ","
                              << windDir9am_pair.second << ","
                              << windDir3pm_pair.first << ","
                              << windDir3pm_pair.second << ","
                              << windSpeed9am << ","
                              << windSpeed3pm << ","
                              << humidity9am << ","
                              << humidity3pm << ","
                              << pressure9am << ","
                              << pressure3pm << ","
                              << cloud3pm << ","
                              << temp9am << ","
                              << diferencaTarde << ","
                              << diferencaManha << ","
                              << diff_temp << ","
                              << diff_press << ","
                              << (rainToday == "Yes" ? 1 : 0) << "\n";
            else if (tipo == 4 || tipo == 5 || tipo == 6 || tipo == 7 || tipo ==8)
                arquivo_saida << data_formatada_x << ","
                              << data_formatada_y << ","
                              << rainfall << ","
                              << sunshine << ","
                              << windGustDir_pair.first << ","
                              << windGustSpeed << ","
                              << windDir9am_pair.first << ","
                              << windDir9am_pair.second << ","
                              << windDir3pm_pair.first << ","
                              << windDir3pm_pair.second << ","
                              << windSpeed3pm << ","
                              << humidity9am << ","
                              << humidity3pm << ","
                              << pressure9am << ","
                              << pressure3pm << ","
                              << cloud3pm << ","
                              << diferencaTarde << ","
                              << diferencaManha << ","
                              << diff_temp << ","
                              << diff_press << "\n";
            labels_saida << (rainTomorrow == "Yes" ? 1 : 0) << "\n";
        }
        count_year_days++; // incremento único por linha, independente de ser pulada ou não
    }

    arquivo_saida.close();
    labels_saida.close();
}

/*
    _tratamento_ dos dados do csv. Linhas com "NA" sao removidas, datas viram um par ordenado (x, y)
    para representar a data em um formato circular, e as direcoes do vento sao convertidas em
    coordenadas circulares (x, y).
    TIPO 1 => SEM FEATURES DERIVADAS
    TIPO 2 => COM FEATURES DERIVADAS
    TIPO 3 => COM FEATURES DERIVADAS PODADO 1
    TIPO 4 => COM FEATURES DERIVADAS PODADO 2
    TIPO 5 => IGUAL AO TIPO 4, MAS TRATANDO DADOS DO SMOTE 0.4
    TIPO 6 => IGUAL AO TIPO 4, MAS TRATANDO DADOS DO SMOTE 0.5
    TIPO 7 => IGUAL AO TIPO 4, MAS TRATANDO DADOS DO SMOTE 0.75
    TIPO 8 => IGUAL AO TIPO 4, MAS TRATANDO DADOS DO SMOTE 1.0
*/
int main(int argc, char *argv[])
{
    int tipo = 0;
    if (argc > 1) {
        tipo = stoi(argv[1]);
    } else {
        cout << "Uso: " << argv[0] << " <tipo: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8>" << endl;
        return 1;
    }

    if (tipo == 1) {
        ofstream treinamento_saida("dados_sem_feature/treinamento_tratado_sem.csv");
        treinamento_saida << "Date_x,Date_y,MinTemp,MaxTemp,RainFall,Evaporation,Sunshine,WindGustDir_x,WindGustDir_y,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday\n";
        ofstream labels_treinamento_saida("dados_sem_feature/labels_treinamento_sem.csv");
        ofstream validacao_saida("dados_sem_feature/validacao_tratado_sem.csv");
        validacao_saida << "Date_x,Date_y,MinTemp,MaxTemp,RainFall,Evaporation,Sunshine,WindGustDir_x,WindGustDir_y,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday\n";
        ofstream labels_validacao_saida("dados_sem_feature/labels_validacao_sem.csv");
        _tratamento_("treinamento.csv", treinamento_saida, labels_treinamento_saida, tipo);
        _tratamento_("validacao.csv", validacao_saida, labels_validacao_saida, tipo);

    } else if (tipo == 2) {
        ofstream treinamento_saida("dados_com_feature/treinamento_tratado_com.csv");
        treinamento_saida << "Date_x,Date_y,MinTemp,MaxTemp,RainFall,Evaporation,Sunshine,WindGustDir_x,WindGustDir_y,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am,RainToday\n";
        ofstream labels_treinamento_saida("dados_com_feature/labels_treinamento_com.csv");
        ofstream validacao_saida("dados_com_feature/validacao_tratado_com.csv");
        validacao_saida << "Date_x,Date_y,MinTemp,MaxTemp,RainFall,Evaporation,Sunshine,WindGustDir_x,WindGustDir_y,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am,RainToday\n";
        ofstream labels_validacao_saida("dados_com_feature/labels_validacao_com.csv");
        _tratamento_("treinamento.csv", treinamento_saida, labels_treinamento_saida, tipo);
        _tratamento_("validacao.csv", validacao_saida, labels_validacao_saida, tipo);

    } else if (tipo == 3) {
        ofstream treinamento_saida("dados_podados_1/treinamento_podado.csv");
        treinamento_saida << "Date_x,Date_y,RainFall,Evaporation,Sunshine,WindGustDir_x,WindGustDir_y,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud3pm,Temp9am,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am,RainToday\n";
        ofstream labels_treinamento_saida("dados_podados_1/labels_treinamento_podado.csv");
        ofstream validacao_saida("dados_podados_1/validacao_podado.csv");
        validacao_saida << "Date_x,Date_y,RainFall,Evaporation,Sunshine,WindGustDir_x,WindGustDir_y,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud3pm,Temp9am,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am,RainToday\n";
        ofstream labels_validacao_saida("dados_podados_1/labels_validacao_podado.csv");
        _tratamento_("treinamento.csv", treinamento_saida, labels_treinamento_saida, tipo);
        _tratamento_("validacao.csv", validacao_saida, labels_validacao_saida, tipo);

    } else if (tipo == 4) {
        ofstream treinamento_saida("dados_podados_2/treinamento_podado_2.csv");
        treinamento_saida << "Date_x,Date_y,RainFall,Sunshine,WindGustDir_x,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud3pm,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am\n";
        ofstream labels_treinamento_saida("dados_podados_2/labels_treinamento_podado_2.csv");
        ofstream validacao_saida("dados_podados_2/validacao_podado_2.csv");
        validacao_saida << "Date_x,Date_y,RainFall,Sunshine,WindGustDir_x,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud3pm,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am\n";
        ofstream labels_validacao_saida("dados_podados_2/labels_validacao_podado_2.csv");
        _tratamento_("treinamento.csv", treinamento_saida, labels_treinamento_saida, tipo);
        _tratamento_("validacao.csv", validacao_saida, labels_validacao_saida, tipo);
	} else if (tipo == 5) {
        ofstream treinamento_saida("dados_smote_04/treinamento_podado_smote.csv");
        treinamento_saida << "Date_x,Date_y,RainFall,Sunshine,WindGustDir_x,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud3pm,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am\n";
        ofstream labels_treinamento_saida("dados_smote_04/labels_treinamento_podado_smote.csv");
        ofstream validacao_saida("dados_smote_04/validacao_podado_smote.csv");
        validacao_saida << "Date_x,Date_y,RainFall,Sunshine,WindGustDir_x,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud3pm,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am\n";
        ofstream labels_validacao_saida("dados_smote_04/labels_validacao_podado_smote.csv");
        _tratamento_("../SMOTE/treinamento_balanceado_04.csv", treinamento_saida, labels_treinamento_saida, tipo);
        _tratamento_("validacao.csv", validacao_saida, labels_validacao_saida, tipo);
        
    } else if (tipo == 6) {
        ofstream treinamento_saida("dados_smote_05/treinamento_podado_smote.csv");
        treinamento_saida << "Date_x,Date_y,RainFall,Sunshine,WindGustDir_x,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud3pm,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am\n";
        ofstream labels_treinamento_saida("dados_smote_05/labels_treinamento_podado_smote.csv");
        ofstream validacao_saida("dados_smote_05/validacao_podado_smote.csv");
        validacao_saida << "Date_x,Date_y,RainFall,Sunshine,WindGustDir_x,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud3pm,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am\n";
        ofstream labels_validacao_saida("dados_smote_05/labels_validacao_podado_smote.csv");
        _tratamento_("../SMOTE/treinamento_balanceado_05.csv", treinamento_saida, labels_treinamento_saida, tipo);
        _tratamento_("validacao.csv", validacao_saida, labels_validacao_saida, tipo);
        
    } else if (tipo == 7) {
        ofstream treinamento_saida("dados_smote_075/treinamento_podado_smote.csv");
        treinamento_saida << "Date_x,Date_y,RainFall,Sunshine,WindGustDir_x,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud3pm,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am\n";
        ofstream labels_treinamento_saida("dados_smote_075/labels_treinamento_podado_smote.csv");
        ofstream validacao_saida("dados_smote_075/validacao_podado_smote.csv");
        validacao_saida << "Date_x,Date_y,RainFall,Sunshine,WindGustDir_x,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud3pm,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am\n";
        ofstream labels_validacao_saida("dados_smote_075/labels_validacao_podado_smote.csv");
        _tratamento_("../SMOTE/treinamento_balanceado_075.csv", treinamento_saida, labels_treinamento_saida, tipo);
        _tratamento_("validacao.csv", validacao_saida, labels_validacao_saida, tipo);
        
    } else if (tipo == 8) {
        ofstream treinamento_saida("dados_smote_1/treinamento_podado_smote.csv");
        treinamento_saida << "Date_x,Date_y,RainFall,Sunshine,WindGustDir_x,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud3pm,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am\n";
        ofstream labels_treinamento_saida("dados_smote_1/labels_treinamento_podado_smote.csv");
        ofstream validacao_saida("dados_smote_1/validacao_podado_smote.csv");
        validacao_saida << "Date_x,Date_y,RainFall,Sunshine,WindGustDir_x,WindGustSpeed,WindDir9am_x,WindDir9am_y,WindDir3pm_x,WindDir3pm_y,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud3pm,dewDiffTemp3pm,dewDiffTemp9am,diffTemp3pm9am,diffPress3pm9am\n";
        ofstream labels_validacao_saida("dados_smote_1/labels_validacao_podado_smote.csv");
        _tratamento_("../SMOTE/treinamento_balanceado_1_0.csv", treinamento_saida, labels_treinamento_saida, tipo);
        _tratamento_("validacao.csv", validacao_saida, labels_validacao_saida, tipo);
        
    }
    else {
        cout << "Tipo inválido. Selecione 1, 2, 3, 4 ou 5";
        cout << "Tipo inválido. Selecione 1, 2, 3 ou 4." << endl;
        return -1;
    }

    return 0;
}

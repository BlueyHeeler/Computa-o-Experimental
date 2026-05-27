#include <iostream>
#include <map>
#include <fstream>
#include <string>
#include <cmath>
#include "csv.h"

using namespace std;

// Função auxiliar para obter o quadrante a partir de uma direção do vento
string getQuadrant(const string& direction) {
    if (direction == "N" || direction == "NNE" || direction == "NE" || direction == "ENE") {
        return "NE";
    } else if (direction == "E" || direction == "ESE" || direction == "SE" || direction == "SSE") {
        return "SE";
    } else if (direction == "S" || direction == "SSW" || direction == "SW" || direction == "WSW") {
        return "SW";
    } else if (direction == "W" || direction == "WNW" || direction == "NW" || direction == "NNW") {
        return "NW";
    }
    return "Unknown";
}

int main(int argc, char* argv[])
{
	string arquivo_entrada;
	if(argc > 1)
		arquivo_entrada = argv[1];
	else
	{
		cout << "Uso: " << argv[0] << "arquivo_entrada.csv" << endl;
		return -1;
	}
	io::CSVReader<22> in(arquivo_entrada);
    in.read_header(io::ignore_extra_column, "Date", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", 
                   "Sunshine", "WindGustDir", "WindGustSpeed", "WindDir9am", 
                   "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", 
                   "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", 
                   "Cloud3pm", "Temp9am", "Temp3pm", "RainToday", "RainTomorrow");
                   
	map<string, int> windQuadrantMorningRain{{"NE", 0}, {"SE", 0}, {"SW", 0}, {"NW", 0}};
	map<string, int> windQuadrantAfternoonRain{{"NE", 0}, {"SE", 0}, {"SW", 0}, {"NW", 0}};
	map<string, int> windQuadrantMorningNoRain{{"NE", 0}, {"SE", 0}, {"SW", 0}, {"NW", 0}};
	map<string, int> windQuadrantAfternoonNoRain{{"NE", 0}, {"SE", 0}, {"SW", 0}, {"NW", 0}};
	
	// Variáveis para dias com chuva
	double total_minTemp_chuva = 0, total_maxTemp_chuva = 0, total_rainfall_chuva = 0, total_evaporation_chuva = 0, total_sunshine_chuva = 0, total_windGustSpeed_chuva = 0, total_windSpeed9am_chuva = 0, total_windSpeed3pm_chuva = 0, total_humidity9am_chuva = 0, total_humidity3pm_chuva = 0, total_pressure9am_chuva = 0, total_pressure3pm_chuva = 0, total_cloud9am_chuva = 0, total_cloud3pm_chuva = 0, total_temp9am_chuva = 0, total_temp3pm_chuva = 0;
	double total_diferenca_chuva = 0;
	double total_diferenca_manha_chuva = 0;
	double total_diff_temp_chuva = 0;
	double total_diff_press_chuva = 0;
	int count_chuva = 0;
	int count_chuva_amanha_se_hoje_chuva = 0;

	// Variáveis para dias sem chuva
	double total_minTemp_sem_chuva = 0, total_maxTemp_sem_chuva = 0, total_rainfall_sem_chuva = 0, total_evaporation_sem_chuva = 0, total_sunshine_sem_chuva = 0, total_windGustSpeed_sem_chuva = 0, total_windSpeed9am_sem_chuva = 0, total_windSpeed3pm_sem_chuva = 0, total_humidity9am_sem_chuva = 0, total_humidity3pm_sem_chuva = 0, total_pressure9am_sem_chuva = 0, total_pressure3pm_sem_chuva = 0, total_cloud9am_sem_chuva = 0, total_cloud3pm_sem_chuva = 0, total_temp9am_sem_chuva = 0, total_temp3pm_sem_chuva = 0;
	double total_diferenca_sem_chuva = 0;
	double total_diferenca_manha_sem_chuva = 0;
	double total_diff_temp_sem_chuva = 0;
	double total_diff_press_sem_chuva = 0;
	int count_sem_chuva = 0;
	int count_chuva_amanha_se_hoje_sem_chuva = 0;
                   
	string date, minTemp, maxTemp, rainfall, evaporation, sunshine, windGustDir, windGustSpeed, 
           windDir9am, windDir3pm, windSpeed9am, windSpeed3pm, humidity9am, humidity3pm, 
           pressure9am, pressure3pm, cloud9am, cloud3pm, temp9am, temp3pm, rainToday, rainTomorrow;
           
	while(in.read_row(date, minTemp, maxTemp, rainfall, evaporation, sunshine, 
                      windGustDir, windGustSpeed, windDir9am, windDir3pm, 
                      windSpeed9am, windSpeed3pm, humidity9am, humidity3pm, 
                      pressure9am, pressure3pm, cloud9am, cloud3pm, 
                      temp9am, temp3pm, rainToday, rainTomorrow))        
	{
		if(date == "NA" || minTemp == "NA" || maxTemp == "NA" || rainfall == "NA" || 
            evaporation == "NA" || sunshine == "NA" || windGustDir == "NA" || windGustSpeed == "NA" || 
            windDir9am == "NA" || windDir3pm == "NA" || windSpeed9am == "NA" || windSpeed3pm == "NA" || 
            humidity9am == "NA" || humidity3pm == "NA" || pressure9am == "NA" || pressure3pm == "NA" || 
            cloud9am == "NA" || cloud3pm == "NA" || temp9am == "NA" || temp3pm == "NA" || 
            rainToday == "NA" || rainTomorrow == "NA" || date.substr(5, 5) == "02-29")
		{
			continue;
		}
		else{
			double temperaturaAtualTarde = stod(temp3pm);
			double umidadeTarde = stod(humidity3pm);
			double a = 17.27;
			double b = 237.7;
			double alphaTarde = ((a * temperaturaAtualTarde) / (b + temperaturaAtualTarde)) + log(umidadeTarde / 100.0);
			double pontoOrvalhoTarde = (b * alphaTarde) / (a - alphaTarde);
			double diferencaTarde = temperaturaAtualTarde - pontoOrvalhoTarde;

			double temperaturaAtualManha = stod(temp9am);
			double umidadeManha = stod(humidity9am);
			double alphaManha = ((a * temperaturaAtualManha) / (b + temperaturaAtualManha)) + log(umidadeManha / 100.0);
			double pontoOrvalhoManha = (b * alphaManha) / (a - alphaManha);
			double diferencaManha = temperaturaAtualManha - pontoOrvalhoManha;

			double diff_temp = stod(temp3pm) - stod(temp9am);
			double diff_press = stod(pressure3pm) - stod(pressure9am);

			if(rainToday == "Yes")
			{
				total_minTemp_chuva += stod(minTemp);
				total_maxTemp_chuva += stod(maxTemp);
				total_rainfall_chuva += stod(rainfall);
				total_evaporation_chuva += stod(evaporation);
				total_sunshine_chuva += stod(sunshine);
				total_windGustSpeed_chuva += stod(windGustSpeed);
				total_windSpeed9am_chuva += stod(windSpeed9am);
				total_windSpeed3pm_chuva += stod(windSpeed3pm);
				total_humidity9am_chuva += stod(humidity9am);
				total_humidity3pm_chuva += stod(humidity3pm);
				total_pressure9am_chuva += stod(pressure9am);
				total_pressure3pm_chuva += stod(pressure3pm);
				total_cloud9am_chuva += stod(cloud9am);
				total_cloud3pm_chuva += stod(cloud3pm);
				total_temp9am_chuva += stod(temp9am);
				total_temp3pm_chuva += stod(temp3pm);

				total_diferenca_chuva += diferencaTarde;
				total_diferenca_manha_chuva += diferencaManha;
				total_diff_temp_chuva += diff_temp;
				total_diff_press_chuva += diff_press;
				count_chuva++;
				if (rainTomorrow == "Yes") {
					count_chuva_amanha_se_hoje_chuva++;
				}
				windQuadrantMorningRain[getQuadrant(windDir9am)] += 1;
				windQuadrantAfternoonRain[getQuadrant(windDir3pm)] += 1;
			}
			else
			{
				total_minTemp_sem_chuva += stod(minTemp);
				total_maxTemp_sem_chuva += stod(maxTemp);
				total_rainfall_sem_chuva += stod(rainfall);
				total_evaporation_sem_chuva += stod(evaporation);
				total_sunshine_sem_chuva += stod(sunshine);
				total_windGustSpeed_sem_chuva += stod(windGustSpeed);
				total_windSpeed9am_sem_chuva += stod(windSpeed9am);
				total_windSpeed3pm_sem_chuva += stod(windSpeed3pm);
				total_humidity9am_sem_chuva += stod(humidity9am);
				total_humidity3pm_sem_chuva += stod(humidity3pm);
				total_pressure9am_sem_chuva += stod(pressure9am);
				total_pressure3pm_sem_chuva += stod(pressure3pm);
				total_cloud9am_sem_chuva += stod(cloud9am);
				total_cloud3pm_sem_chuva += stod(cloud3pm);
				total_temp9am_sem_chuva += stod(temp9am);
				total_temp3pm_sem_chuva += stod(temp3pm);

				total_diferenca_sem_chuva += diferencaTarde;
				total_diferenca_manha_sem_chuva += diferencaManha;
				total_diff_temp_sem_chuva += diff_temp;
				total_diff_press_sem_chuva += diff_press;
				count_sem_chuva++;
				if (rainTomorrow == "Yes") {
					count_chuva_amanha_se_hoje_sem_chuva++;
				}
				windQuadrantMorningNoRain[getQuadrant(windDir9am)] += 1;
				windQuadrantAfternoonNoRain[getQuadrant(windDir3pm)] += 1;
			}
		}
	}

	cout << "--- Análise Dias de Chuva ---" << endl;
	cout << "Total de dias com chuva: " << count_chuva << endl;
	if(count_chuva > 0) {
		cout << "Média MinTemp: " << total_minTemp_chuva / count_chuva << endl;
		cout << "Média MaxTemp: " << total_maxTemp_chuva / count_chuva << endl;
		cout << "Média Rainfall: " << total_rainfall_chuva / count_chuva << endl;
		cout << "Média Evaporation: " << total_evaporation_chuva / count_chuva << endl;
		cout << "Média Sunshine: " << total_sunshine_chuva / count_chuva << endl;
		cout << "Média WindGustSpeed: " << total_windGustSpeed_chuva / count_chuva << endl;
		cout << "Média WindSpeed9am: " << total_windSpeed9am_chuva / count_chuva << endl;
		cout << "Média WindSpeed3pm: " << total_windSpeed3pm_chuva / count_chuva << endl;
		cout << "Média Humidity9am: " << total_humidity9am_chuva / count_chuva << endl;
		cout << "Média Humidity3pm: " << total_humidity3pm_chuva / count_chuva << endl;
		cout << "Média Pressure9am: " << total_pressure9am_chuva / count_chuva << endl;
		cout << "Média Pressure3pm: " << total_pressure3pm_chuva / count_chuva << endl;
		cout << "Média Cloud9am: " << total_cloud9am_chuva / count_chuva << endl;
		cout << "Média Cloud3pm: " << total_cloud3pm_chuva / count_chuva << endl;
		cout << "Média Temp9am: " << total_temp9am_chuva / count_chuva << endl;
		cout << "Média Temp3pm: " << total_temp3pm_chuva / count_chuva << endl;
		cout << "Média da diferença Temp-Orvalho (Manhã): " << total_diferenca_manha_chuva / count_chuva << endl;
		cout << "Média da diferença Temp-Orvalho (Tarde): " << total_diferenca_chuva / count_chuva << endl;
		cout << "Média da diferença de Temperatura (3pm-9am): " << total_diff_temp_chuva / count_chuva << endl;
		cout << "Média da diferença de Pressão (3pm-9am): " << total_diff_press_chuva / count_chuva << endl;
		cout << "Média de chuva amanhã dado que choveu hoje: "
			 << static_cast<double>(count_chuva_amanha_se_hoje_chuva) / count_chuva << endl;
	}
	cout << "Direção do vento pela manhã:" << endl;
	for(auto const& [key, val] : windQuadrantMorningRain)
	{
		if(val > 0)
			cout << key << " : " << val << endl;
	}
	cout << "Direção do vento pela tarde:" << endl;
	for(auto const& [key, val] : windQuadrantAfternoonRain)
	{
		if(val > 0)
			cout << key << " : " << val << endl;
	}

	cout << "\n--- Análise Dias Sem Chuva ---" << endl;
	cout << "Total de dias sem chuva: " << count_sem_chuva << endl;
	if(count_sem_chuva > 0) {
		cout << "Média MinTemp: " << total_minTemp_sem_chuva / count_sem_chuva << endl;
		cout << "Média MaxTemp: " << total_maxTemp_sem_chuva / count_sem_chuva << endl;
		cout << "Média Rainfall: " << total_rainfall_sem_chuva / count_sem_chuva << endl;
		cout << "Média Evaporation: " << total_evaporation_sem_chuva / count_sem_chuva << endl;
		cout << "Média Sunshine: " << total_sunshine_sem_chuva / count_sem_chuva << endl;
		cout << "Média WindGustSpeed: " << total_windGustSpeed_sem_chuva / count_sem_chuva << endl;
		cout << "Média WindSpeed9am: " << total_windSpeed9am_sem_chuva / count_sem_chuva << endl;
		cout << "Média WindSpeed3pm: " << total_windSpeed3pm_sem_chuva / count_sem_chuva << endl;
		cout << "Média Humidity9am: " << total_humidity9am_sem_chuva / count_sem_chuva << endl;
		cout << "Média Humidity3pm: " << total_humidity3pm_sem_chuva / count_sem_chuva << endl;
		cout << "Média Pressure9am: " << total_pressure9am_sem_chuva / count_sem_chuva << endl;
		cout << "Média Pressure3pm: " << total_pressure3pm_sem_chuva / count_sem_chuva << endl;
		cout << "Média Cloud9am: " << total_cloud9am_sem_chuva / count_sem_chuva << endl;
		cout << "Média Cloud3pm: " << total_cloud3pm_sem_chuva / count_sem_chuva << endl;
		cout << "Média Temp9am: " << total_temp9am_sem_chuva / count_sem_chuva << endl;
		cout << "Média Temp3pm: " << total_temp3pm_sem_chuva / count_sem_chuva << endl;
		cout << "Média da diferença Temp-Orvalho (Manhã): " << total_diferenca_manha_sem_chuva / count_sem_chuva << endl;
		cout << "Média da diferença Temp-Orvalho (Tarde): " << total_diferenca_sem_chuva / count_sem_chuva << endl;
		cout << "Média da diferença de Temperatura (3pm-9am): " << total_diff_temp_sem_chuva / count_sem_chuva << endl;
		cout << "Média da diferença de Pressão (3pm-9am): " << total_diff_press_sem_chuva / count_sem_chuva << endl;
		cout << "Média de chuva amanhã dado que não choveu hoje: "
			 << static_cast<double>(count_chuva_amanha_se_hoje_sem_chuva) / count_sem_chuva << endl;
	}
	cout << "Direção do vento pela manhã:" << endl;
	for(auto const& [key, val] : windQuadrantMorningNoRain)
	{
		if(val > 0)
			cout << key << " : " << val << endl;
	}
	cout << "Direção do vento pela tarde:" << endl;
	for(auto const& [key, val] : windQuadrantAfternoonNoRain)
	{
		if(val > 0)
			cout << key << " : " << val << endl;
	}

	return 0;
}

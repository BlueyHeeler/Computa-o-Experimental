import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

def balancear_e_reverter(df, target_col='RainTomorrow', proporcoes=[0.4, 0.5, 0.75, 1.0]):
    """
    df: Seu DataFrame original
    target_col: Nome da coluna que queremos balancear
    proporcoes: Lista com as proporções minoritário/majoritário desejadas.
                (ex: 1.0 = 50%/50%, 0.5 = 33%/66%)
    """
    df_processed = df.copy()
    
    # 1. Definir os tipos de colunas
    date_col = 'Date'
    cat_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    num_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 
                'Cloud3pm', 'Temp9am', 'Temp3pm']
    
    # 2. Codificar a Data para numérico (Ordinal)
    df_processed[date_col] = pd.to_datetime(df_processed[date_col]).apply(lambda x: x.toordinal())
    
    # 3. Label Encoding para categorias e para a variável alvo (RainTomorrow)
    le_dict = {}
    for col in cat_cols + [target_col]:
        le = LabelEncoder()
        # Converte para string para evitar erros com possíveis valores nulos
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        le_dict[col] = le
        
    # 4. Normalização das variáveis numéricas e da data com MinMaxScaler
    scaler = MinMaxScaler()
    cols_to_scale = num_cols + [date_col]
    df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
    
    # Separar X (features) e y (target)
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    feature_cols = X.columns.tolist()
    datasets_finais = {}
    
    # 5. Aplicar SMOTE iterando pelas proporções desejadas
    for ratio in proporcoes:
        print(f"\nAplicando SMOTE para a proporção: {ratio}")
        try:
            # O sampling_strategy dita a nova proporção da classe minoritária
            smote = SMOTE(sampling_strategy=ratio, random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            
            # Reconstruir o DataFrame pós-SMOTE
            df_res = pd.DataFrame(X_res, columns=feature_cols)
            df_res[target_col] = y_res
            
            # 6. REVERTER A NORMALIZAÇÃO
            df_res[cols_to_scale] = scaler.inverse_transform(df_res[cols_to_scale])
            
            # 7. REVERTER A DATA
            # Arredondamos pois o SMOTE gera floats (ex: data 733408.5 vira 733409)
            df_res[date_col] = df_res[date_col].round().astype(int)
            df_res[date_col] = df_res[date_col].apply(lambda x: pd.Timestamp.fromordinal(x).strftime('%Y-%m-%d'))
            
            # 8. REVERTER AS CATEGORIAS
            for col in cat_cols + [target_col]:
                # Arredondar os valores interpolados pelo SMOTE para o inteiro mais próximo
                df_res[col] = df_res[col].round().astype(int)
                
                # Garantir que o arredondamento não ultrapasse os limites das classes originais
                max_class_val = len(le_dict[col].classes_) - 1
                df_res[col] = df_res[col].clip(0, max_class_val)
                
                # Reverter para o texto original
                df_res[col] = le_dict[col].inverse_transform(df_res[col])
                
            datasets_finais[f"Dataset_Proporcao_{ratio}"] = df_res
            print(f"Sucesso! Nova distribuição de {target_col}:")
            print(df_res[target_col].value_counts())
            
        except Exception as e:
            print(f"Erro ao aplicar a proporção {ratio}: {e}")
            
    return datasets_finais

# ==========================================
# EXECUÇÃO DO SEU ARQUIVO treinamento.csv
# ==========================================

# 1. Lê o arquivo
df_original = pd.read_csv('treinamento.csv')

# -> NOVIDADE: Remove todas as linhas que contenham qualquer valor NaN
df_original = df_original.dropna()

# 2. Executa a função pedindo 3 datasets diferentes: 
# 0.5 (1/2), 0.75 (3/4) e 1.0 (Metade/Metade)
datasets_gerados = balancear_e_reverter(df_original, target_col='RainTomorrow', proporcoes=[0.4, 0.5, 0.75, 1.0])

# 3. Salva os novos arquivos balanceados em CSV
if "Dataset_Proporcao_0.4" in datasets_gerados:
    datasets_gerados["Dataset_Proporcao_0.4"].to_csv("treinamento_balanceado_04.csv", index=False)
    
if "Dataset_Proporcao_0.5" in datasets_gerados:
    datasets_gerados["Dataset_Proporcao_0.5"].to_csv("treinamento_balanceado_05.csv", index=False)
    
if "Dataset_Proporcao_0.75" in datasets_gerados:
    datasets_gerados["Dataset_Proporcao_0.75"].to_csv("treinamento_balanceado_075.csv", index=False)

if "Dataset_Proporcao_1.0" in datasets_gerados:
    datasets_gerados["Dataset_Proporcao_1.0"].to_csv("treinamento_balanceado_1_0.csv", index=False)

print("\nArquivos CSV gerados com sucesso e sem valores NaN!")

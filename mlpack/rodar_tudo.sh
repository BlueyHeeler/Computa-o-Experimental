#!/bin/bash

# Cria a pasta para organizar os resultados
mkdir -p resultados

# Explicação do xargs abaixo:
# -I {} substitui o {} pelo número da vez
# -P 4 define o limite máximo de 4 processos simultâneos
seq 1 12 | xargs -I {} -P 4 sh -c '
    echo "Iniciando n = {}..."
    ./importancia_media {} > "resultados/saida_{}.txt" 2>&1
    echo "Concluído n = {}!"
'

echo "--- TODOS OS 12 PROCESSOS FORAM CONCLUÍDOS! ---"

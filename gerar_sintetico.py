#!/usr/bin/env python3
"""
Gera dados sinteticos de dias de chuva via copula gaussiana.
Base:      dataset PODADO 2 (20 features, melhor configuracao).
Quantidade: suficiente para balancear as classes 1:1.
Saida:     mlpack/dados_augmented/
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

SEED = 42
TRAIN_FEATURES = "mlpack/dados_podados_2/treinamento_podado_2.csv"
TRAIN_LABELS   = "mlpack/dados_podados_2/labels_treinamento_podado_2.csv"
OUT_DIR        = "mlpack/dados_augmented"


def gaussian_copula_sample(
    X_ref: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Estima uma copula gaussiana a partir de X_ref e amostra n_samples pontos.

    Passos:
      1. Rank transform -> uniforme em (0,1)
      2. Probit          -> normal padrao
      3. Correlacao      -> normal multivariada com estrutura aprendida
      4. Normal -> uniforme -> espaco original via quantil empirico
    """
    n, d = X_ref.shape
    X = X_ref.values.astype(float)

    # Passo 1 — rank transform, evitando bordas 0 e 1
    U = np.zeros((n, d))
    for j in range(d):
        U[:, j] = stats.rankdata(X[:, j]) / (n + 1)

    # Passo 2 — probit
    Z = stats.norm.ppf(U)

    # Passo 3 — matriz de correlacao (garante positiva semi-definida)
    R = np.corrcoef(Z.T)
    R = (R + R.T) / 2
    min_eig = np.linalg.eigvalsh(R).min()
    if min_eig < 1e-8:
        R += (-min_eig + 1e-8) * np.eye(d)

    Z_new = rng.multivariate_normal(np.zeros(d), R, size=n_samples)

    # Passo 4 — inverter via CDF empirica (interpolacao)
    U_new = np.clip(stats.norm.cdf(Z_new), 1e-10, 1 - 1e-10)
    X_new = np.zeros((n_samples, d))
    for j in range(d):
        col_sorted = np.sort(X[:, j])
        quantiles  = np.linspace(0, 1, n)
        X_new[:, j] = np.interp(U_new[:, j], quantiles, col_sorted)

    return pd.DataFrame(X_new, columns=X_ref.columns)


def main() -> None:
    rng = np.random.default_rng(SEED)

    X = pd.read_csv(TRAIN_FEATURES)
    y = pd.read_csv(TRAIN_LABELS, header=None, names=["label"])

    n_rain    = int((y["label"] == 1).sum())
    n_no_rain = int((y["label"] == 0).sum())
    n_synth   = n_no_rain - n_rain  # balancear para 1:1

    print(f"Dias com chuva  (real):  {n_rain}")
    print(f"Dias sem chuva  (real):  {n_no_rain}")
    print(f"Amostras sinteticas:     {n_synth}")
    print(f"Features:                {X.shape[1]}")

    X_rain = X[y["label"].values == 1].reset_index(drop=True)

    print("\nGerando amostras via copula gaussiana...")
    X_synth = gaussian_copula_sample(X_rain, n_synth, rng)
    y_synth = pd.DataFrame({"label": np.ones(n_synth, dtype=int)})

    X_aug = pd.concat([X, X_synth], ignore_index=True)
    y_aug = pd.concat([y, y_synth], ignore_index=True)

    # Embaralhar para nao deixar todos os sinteticos no final
    perm  = rng.permutation(len(X_aug))
    X_aug = X_aug.iloc[perm].reset_index(drop=True)
    y_aug = y_aug.iloc[perm].reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    X_aug.to_csv(f"{OUT_DIR}/treinamento_augmented.csv", index=False)
    y_aug.to_csv(f"{OUT_DIR}/labels_treinamento_augmented.csv", index=False, header=False)

    dist = y_aug["label"].value_counts().sort_index()
    print(f"\nSalvo em '{OUT_DIR}/'")
    print(f"Total de amostras: {len(X_aug)}")
    print(f"  Sem chuva (0): {dist[0]}")
    print(f"  Com chuva (1): {dist[1]}")

    # Verificacao de sanidade: medias deveriam ser parecidas
    print("\n--- Verificacao: medias reais vs sinteticas (dias de chuva) ---")
    for col in X.columns:
        mu_real  = X_rain[col].mean()
        std_real = X_rain[col].std() + 1e-9
        mu_synth = X_synth[col].mean()
        diff_pct = abs(mu_synth - mu_real) / std_real * 100
        flag = "  << DIVERGENTE" if diff_pct > 15 else ""
        print(f"  {col:<22}  real={mu_real:8.3f}  synth={mu_synth:8.3f}  diff={diff_pct:5.1f}%{flag}")


if __name__ == "__main__":
    main()

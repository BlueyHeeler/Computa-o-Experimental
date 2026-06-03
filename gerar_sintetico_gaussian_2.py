#!/usr/bin/env python3
"""
Gera dados sintéticos de dias de chuva via cópula gaussiana.
Base:      dataset PODADO 2 (20 features, melhor configuração).
Quantidade: controlada por SYNTH_RATIO (fração do desbalanceamento a preencher).
Saída:     csv/dados_augmented_gaussian/

SYNTH_RATIO — quanto do gap entre classes se deseja preencher com sintéticos:
  1.0  → balanceamento 1:1 completo  (comportamento original)
  0.5  → preenche metade do gap       (ex: 1000 rain / 3000 no-rain → +1000 sintéticos)
  0.25 → preenche 25 % do gap
  0.0  → nenhum sintético gerado
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
import os

# ── Configurações padrão ────────────────────────────────────────────────────
SEED = 42

TRAIN_FEATURES = "csv/dados_com_feature/treinamento_tratado_com.csv"
TRAIN_LABELS   = "csv/dados_com_feature/labels_treinamento_com.csv"
OUT_DIR        = "csv/dados_augmented_gaussian"

# Colunas do PODADO 2 — subconjunto que será usado no treinamento do RF
PODADO2_COLS = [
    "Date_x", "Date_y",
    "RainFall", "Sunshine",
    "WindGustDir_x", "WindGustSpeed",
    "WindDir9am_x", "WindDir9am_y",
    "WindDir3pm_x", "WindDir3pm_y",
    "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Cloud3pm",
    "dewDiffTemp3pm", "dewDiffTemp9am",
    "diffTemp3pm9am", "diffPress3pm9am",
]

# ── Cópula gaussiana ─────────────────────────────────────────────────────────

def gaussian_copula_sample(
    X_ref: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Estima uma cópula gaussiana a partir de X_ref e amostra n_samples pontos.

    Passos:
      1. Rank transform  → uniforme em (0, 1)
      2. Probit           → normal padrão
      3. Correlação       → normal multivariada com estrutura aprendida
      4. Normal → uniforme → espaço original via quantil empírico
    """
    n, d = X_ref.shape
    X = X_ref.values.astype(float)

    # Passo 1 — rank transform, evitando bordas 0 e 1
    U = np.zeros((n, d))
    for j in range(d):
        # Correção: usa (i - 0.5)/n para evitar extrapolação nas caudas
        U[:, j] = (stats.rankdata(X[:, j]) - 0.5) / n

    # Passo 2 — probit
    Z = stats.norm.ppf(U)

    # Passo 3 — matriz de correlação (garante positiva semi-definida)
    R = np.corrcoef(Z.T)
    R = (R + R.T) / 2
    min_eig = np.linalg.eigvalsh(R).min()
    if min_eig < 1e-8:
        R += (-min_eig + 1e-8) * np.eye(d)

    Z_new = rng.multivariate_normal(np.zeros(d), R, size=n_samples)

    # Passo 4 — inverter via CDF empírica (interpolação)
    U_new = np.clip(stats.norm.cdf(Z_new), 1e-10, 1 - 1e-10)
    X_new = np.zeros((n_samples, d))
    for j in range(d):
        col_sorted = np.sort(X[:, j])
        # Quantis centrados: evita extrapolar além do min/max observado
        quantiles  = (np.arange(1, n + 1) - 0.5) / n
        X_new[:, j] = np.interp(U_new[:, j], quantiles, col_sorted)

    return pd.DataFrame(X_new, columns=X_ref.columns)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera dados sintéticos via cópula gaussiana.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        metavar="RATIO",
        help=(
            "Fração do desbalanceamento a preencher com amostras sintéticas. "
            "1.0 = balanceamento 1:1 completo; "
            "0.5 = preenche metade do gap; "
            "0.0 = nenhum sintético."
        ),
    )
    parser.add_argument(
        "--features", default=TRAIN_FEATURES, help="Caminho do CSV de features."
    )
    parser.add_argument(
        "--labels", default=TRAIN_LABELS, help="Caminho do CSV de labels."
    )
    parser.add_argument(
        "--out-dir", default=OUT_DIR, help="Diretório de saída."
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Semente aleatória."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validação do ratio ────────────────────────────────────────────────
    if not (0.0 <= args.ratio <= 1.0):
        raise ValueError(
            f"--ratio deve estar entre 0.0 e 1.0, mas recebeu {args.ratio}."
        )

    rng = np.random.default_rng(args.seed)

    X_full = pd.read_csv(args.features)
    y      = pd.read_csv(args.labels, header=None, names=["label"])

    # RainToday não faz parte do PODADO 2; remover antes de aprender a cópula
    X_full = X_full.drop(columns=["RainToday"], errors="ignore")

    n_rain    = int((y["label"] == 1).sum())
    n_no_rain = int((y["label"] == 0).sum())
    gap       = n_no_rain - n_rain            # diferença entre classes

    # ── Quantidade de sintéticos controlada por --ratio ───────────────────
    # ratio=1.0 → n_synth = gap   (1:1 perfeito)
    # ratio=0.5 → n_synth = gap/2
    # ratio=0.0 → n_synth = 0
    n_synth = max(0, round(gap * args.ratio))

    print(f"Dias com chuva  (real):    {n_rain}")
    print(f"Dias sem chuva  (real):    {n_no_rain}")
    print(f"Gap entre classes:         {gap}")
    print(f"Ratio selecionado:         {args.ratio:.2f}")
    print(f"Amostras sintéticas:       {n_synth}")
    print(
        f"Proporção após augment:    "
        f"{n_rain + n_synth} rain / {n_no_rain} no-rain "
        f"({(n_rain + n_synth) / n_no_rain:.2f}:1)"
    )
    print(f"Features (cópula):         {X_full.shape[1]}  (todos os pares x/y intactos)")

    X_real = X_full[PODADO2_COLS]
    y_aug  = y.copy()
    X_aug  = X_real.copy()

    if n_synth > 0:
        # Aprender a cópula nas features completas (pares direcionais preservados)
        X_rain_full = X_full[y["label"].values == 1].reset_index(drop=True)

        print("\nGerando amostras via cópula gaussiana...")
        X_synth_full = gaussian_copula_sample(X_rain_full, n_synth, rng)
        X_synth      = X_synth_full[PODADO2_COLS]
        y_synth      = pd.DataFrame({"label": np.ones(n_synth, dtype=int)})

        X_aug = pd.concat([X_real, X_synth], ignore_index=True)
        y_aug = pd.concat([y,      y_synth], ignore_index=True)

        # Verificação de sanidade: médias deveriam ser parecidas
        X_rain_podado = X_rain_full[PODADO2_COLS]
        print("\n--- Verificação: médias reais vs sintéticas (dias de chuva, PODADO 2) ---")
        for col in PODADO2_COLS:
            mu_real  = X_rain_podado[col].mean()
            std_real = X_rain_podado[col].std() + 1e-9
            mu_synth = X_synth[col].mean()
            diff_pct = abs(mu_synth - mu_real) / std_real * 100
            flag = "  << DIVERGENTE" if diff_pct > 15 else ""
            print(
                f"  {col:<22}  real={mu_real:8.3f}  "
                f"synth={mu_synth:8.3f}  diff={diff_pct:5.1f}%{flag}"
            )
    else:
        print("\nRatio=0: nenhuma amostra sintética gerada. Salvando dados originais.")

    # Embaralhar para não deixar todos os sintéticos no final
    perm  = rng.permutation(len(X_aug))
    X_aug = X_aug.iloc[perm].reset_index(drop=True)
    y_aug = y_aug.iloc[perm].reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    X_aug.to_csv(f"{args.out_dir}/treinamento_augmented.csv", index=False)
    y_aug.to_csv(
        f"{args.out_dir}/labels_treinamento_augmented.csv",
        index=False,
        header=False,
    )

    dist = y_aug["label"].value_counts().sort_index()
    print(f"\nSalvo em '{args.out_dir}/'")
    print(f"Total de amostras: {len(X_aug)}")
    print(f"  Sem chuva (0): {dist[0]}")
    print(f"  Com chuva (1): {dist[1]}")


if __name__ == "__main__":
    main()

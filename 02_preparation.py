# -*- coding: utf-8 -*-
"""
02_preparation.py
Preparação do dataset para modelagem preditiva da inadimplência de cartões.

Objetivo:
    - Utilizar as variáveis explicativas "cruas" (sem lags)
    - Incluir apenas a defasagem de 1 mês da variável TARGET (TARGET_lag1)
    - Evitar data leakage: para cada mês t, o modelo usa:
        * X_t (variáveis macroeconômicas do próprio mês)
        * y_{t-1} (inadimplência do mês anterior)
        * TARGET_t como variável alvo (y_t)

Saída:
    - prepared/prepared_FULL.parquet
"""

#%% ---------------------- Importações ----------------------
import numpy as np
import pandas as pd
from pathlib import Path

#%% ---------------------- Configuração ---------------------
DATA_DIR = Path("./data")
PREP_DIR = Path("./prepared")
PREP_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "dados_consolidados_macro_credito.parquet"
TARGET     = "inadimpl_cartao_total"
DATE_COL   = "data"

# Mantém colunas com >= 70% de não-nulos (por segurança/flexibilidade)
MIN_NONNA_RATIO = 0.70

#%% ---------------------- Funções auxiliares ---------------
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e organiza o DataFrame:
        - converte a coluna de data
        - ordena cronologicamente
        - filtra colunas com boa completude
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Seleciona colunas com boa completude (na prática, todas do seu dataset)
    keep = [DATE_COL] + [
        c for c in df.columns
        if c != DATE_COL and df[c].notna().mean() >= MIN_NONNA_RATIO
    ]
    df = df[keep]

    # Não aplica forward fill – conforme definido no TCC (sem imputação)
    return df


def _finalize(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """
    Remove linhas com ausências nas features ou na TARGET e
    retorna DataFrame final com colunas na ordem:
        [DATE_COL, TARGET] + feature_cols
    """
    final = df.dropna(subset=[TARGET] + feature_cols).copy()
    final = final[[DATE_COL, TARGET] + feature_cols]
    return final

#%% ---------------------- Pipeline principal ---------------
def prepare_full_dataset() -> pd.DataFrame:
    """
    Prepara o dataset FULL para modelagem, com:
        - variáveis explicativas em nível (sem lags)
        - TARGET_lag1 como variável preditora adicional
    """
    print("\n=== Preparando dataset FULL (variáveis cruas + TARGET_lag1) ===")

    if not INPUT_FILE.exists():
        raise SystemExit(f"Arquivo não encontrado: {INPUT_FILE}")

    # Carrega base consolidada
    base = pd.read_parquet(INPUT_FILE)
    base = _clean_df(base)

    # Preditores numéricos (todas as variáveis numéricas exceto TARGET)
    numeric_cols = base.select_dtypes(include=[np.number]).columns.tolist()
    predictors = [c for c in numeric_cols if c != TARGET]

    # Cria lag de 1 mês APENAS para a variável alvo
    # Para cada data t:
    #   - TARGET (y_t) é o valor a ser previsto
    #   - TARGET_lag1 (y_{t-1}) é usado como preditor
    base[f"{TARGET}_lag1"] = base[TARGET].shift(1)

    # Features = variáveis macro "cruas" + TARGET_lag1
    feature_cols = predictors + [f"{TARGET}_lag1"]

    # Remove primeira linha (que terá TARGET_lag1 = NaN) e qualquer outra linha com NaN
    final = _finalize(base, feature_cols)

    print(f"Observações finais: {len(final)} | Features: {len(feature_cols)}")
    print(f"Primeira data: {final[DATE_COL].min().date()} | Última data: {final[DATE_COL].max().date()}")

    # Salva dataset preparado
    out_path = PREP_DIR / "prepared_FULL.parquet"
    final.to_parquet(out_path, index=False)
    print(f"Dataset preparado salvo em: {out_path}")

    return final

#%% ---------------------- Execução -------------------------
if __name__ == "__main__":
    _ = prepare_full_dataset()

# -*- coding: utf-8 -*-
"""
01_extraction.py
TCC – Risco de Crédito em Cartões × Macroeconomia (BCB/SGS)
Objetivo: APENAS extrair e consolidar séries do BCB/SGS em um único .xlsx/.parquet.
(Plotagens e pré-processamento ficam em 02_preparation.py)
"""
import requests, pandas as pd
from pathlib import Path
from functools import reduce

DATA_DIR = Path("./data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
START = "2015-01-01"; END = "2025-07-01"

SERIES = {
    "selic_mensal": 4390,
    "ibcbr_dessaz": 24364,
    "ibcbr_sem_ajuste": 24363,
    "inadimpl_cartao_total": 25464,
    "ipca_mensal": 433,
    "comprometimento_renda": 29034,
    "endividamento_familias": 29037,
    "inadimplencia_familias": 21082,
}

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json",
                        "User-Agent": "Mozilla/5.0 (SGS-client; TCC Jorge Fumagalli)"})

def _fmt(d): return pd.to_datetime(d).strftime("%d/%m/%Y")
def _fetch(host, codigo, start, end, timeout=60):
    url = f"{host}/dados/serie/bcdata.sgs.{codigo}/dados"
    r = SESSION.get(url, params={"formato": "json", "dataInicial": _fmt(start), "dataFinal": _fmt(end)}, timeout=timeout)
    return r

def get_sgs(codigo, start=START, end=END, timeout=60):
    for host in ["https://api.bcb.gov.br", "https://dadosabertos.bcb.gov.br"]:
        r = _fetch(host, codigo, start, end, timeout)
        if r.status_code == 200 and "json" in (r.headers.get("Content-Type","").lower()):
            data = r.json()
            df = pd.DataFrame(data)
            if df.empty: return pd.DataFrame(columns=["data","valor"])
            df["valor"] = pd.to_numeric(df["valor"].astype(str).str.replace(",", "."), errors="coerce")
            df["data"]  = pd.to_datetime(df["data"], dayfirst=True)
            return df
    raise RuntimeError(f"Falha SGS série {codigo}")

def to_month_start(df):
    df=df.copy(); df["data"]=df["data"].dt.to_period("M").dt.start_time; return df

def main():
    print("="*80); print("01 - EXTRAÇÃO DE SÉRIES (SGS/BCB)"); print("="*80)
    dfs=[]
    for name, code in SERIES.items():
        try:
            df = get_sgs(code, START, END); df = to_month_start(df).rename(columns={"valor": name})
            dfs.append(df[["data", name]])
        except Exception as e:
            print(f"[ERRO] {name}: {e}")
    base = reduce(lambda l, r: pd.merge(l, r, on="data", how="outer"), dfs).sort_values("data").reset_index(drop=True)
    (DATA_DIR/"dados_consolidados_macro_credito.xlsx").write_text("")  # placeholder
    base.to_excel(DATA_DIR/"dados_consolidados_macro_credito.xlsx", index=False)
    base.to_parquet(DATA_DIR/"dados_consolidados_macro_credito.parquet", index=False)
    print("OK -> ./data/")

if __name__=="__main__": main()

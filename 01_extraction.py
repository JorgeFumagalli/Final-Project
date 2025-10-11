# -*- coding: utf-8 -*-
"""
TCC – Risco de Crédito em Cartões × Macroeconomia (BCB/SGS)
Versão Final - Apenas séries confiáveis
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from functools import reduce

# =============================================================================
# Configurações
# =============================================================================

OUTPUT_DIR = Path(r"D:\Degas\Documents\Pós Gradução - Data Science and Analytics\TCC\Saídas")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Séries mensais diretas
SERIES = {
    # Política Monetária
    "selic_mensal": 4390,        # SELIC acumulada no mês (% a.m.)
    
    # Atividade Econômica
    "ibcbr_dessaz": 24364,       # IBC-Br dessazonalizado (índice)
    "ibcbr_sem_ajuste": 24363,   # IBC-Br sem ajuste sazonal (índice)
    
    # Inadimplência de cartão (% do saldo da carteira)
    "inadimpl_cartao_total": 25464,
    
    # Preços
    "ipca_mensal": 433,          # IPCA - variação mensal (% a.m.)
    
    # Situação das Famílias
    "comprometimento_renda_sem_ajuste": 29265,  # Comprometimento de renda das famílias com o serviço da dívida sem ajuste sazonalidade - %
    "comprometimento_renda": 29034,  # Comprometimento de renda das famílias com o serviço da dívida com o Sistema Financeiro Nacional - Com ajuste sazonal
    "endividamento_familias": 29037,  # Endividamento das famílias com SFN (%) - Até 2021
    "inadimplencia_familias": 21082,  # Inadimplência das famílias (%)
}    

START = "2015-01-01"
END   = "2025-07-01"

# =============================================================================
# Sessão HTTP + utilitários
# =============================================================================

SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (SGS-client; TCC Jorge Fumagalli)"
})

def _fmt_bcb_date(d):
    if d is None or d == "":
        return None
    return pd.to_datetime(d).strftime("%d/%m/%Y")

def _fetch_series(host_base, codigo, start, end, timeout=60):
    url = f"{host_base}/dados/serie/bcdata.sgs.{codigo}/dados"
    params = {"formato": "json"}
    if start: params["dataInicial"] = _fmt_bcb_date(start)
    if end:   params["dataFinal"]   = _fmt_bcb_date(end)
    r = SESSION.get(url, params=params, timeout=timeout)
    print(f"[SGS] {r.status_code} | {r.headers.get('Content-Type','')} | {r.url}")
    return r

def get_sgs(codigo, start="2015-01-01", end="2025-07-01", timeout=60):
    """
    Baixa série SGS em DataFrame [data, valor] com:
    - validação de JSON,
    - fallback de host: api.bcb.gov.br -> dadosabertos.bcb.gov.br.
    """
    hosts = ["https://api.bcb.gov.br", "https://dadosabertos.bcb.gov.br"]
    last_diag = None
    for host in hosts:
        r = _fetch_series(host, codigo, start, end, timeout=timeout)
        ctype = (r.headers.get("Content-Type") or "").lower()

        if r.status_code != 200 or "json" not in ctype:
            snippet = (r.text or "")[:300].replace("\n", " ")
            last_diag = f"Status={r.status_code} | CT={ctype} | Body~ {snippet}"
            continue

        try:
            data = r.json()
        except Exception:
            snippet = (r.text or "")[:300].replace("\n", " ")
            last_diag = f"JSON decode falhou | Status={r.status_code} | CT={ctype} | Body~ {snippet}"
            continue

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(columns=["data", "valor"])
        df["valor"] = pd.to_numeric(df["valor"].astype(str).str.replace(",", "."), errors="coerce")
        df["data"]  = pd.to_datetime(df["data"], dayfirst=True)
        return df

    raise RuntimeError(
        f"Falha ao obter JSON para a série {codigo}. Último diagnóstico: {last_diag}"
    )

def to_month_start(df):
    """Converte a data para o primeiro dia do mês (compatível com pandas recente)."""
    df = df.copy()
    df["data"] = df["data"].dt.to_period("M").dt.start_time
    return df

# =============================================================================
# Execução principal
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TCC - Risco de Crédito em Cartões × Macroeconomia")
    print("="*80)
    
    # 1) Baixar todas as séries
    dfs = []
    for colname, code in SERIES.items():
        print(f"\nBaixando {colname} (código {code})...")
        try:
            df = get_sgs(code, start=START, end=END)
        except Exception as e:
            print(f"[ERRO] Série {colname} ({code}) falhou: {e}")
            continue

        if df.empty:
            print(f"[AVISO] Série {colname} ({code}) veio vazia.")
            continue

        df = to_month_start(df).rename(columns={"valor": colname})
        dfs.append(df[["data", colname]])
        print(f"✓ {colname}: {len(df)} observações baixadas")

    base = reduce(lambda l, r: pd.merge(l, r, on="data", how="outer"), dfs)
    base = base.sort_values("data").reset_index(drop=True)

    # 3) Informações sobre disponibilidade de dados
    print("\n" + "="*80)
    print("RESUMO DAS SÉRIES")
    print("="*80)
    
    for col in base.columns:
        if col != "data":
            n_obs = base[col].notna().sum()
            first_date = base.loc[base[col].notna(), "data"].min()
            last_date = base.loc[base[col].notna(), "data"].max()
            print(f"\n{col}:")
            print(f"  Observações: {n_obs}")
            print(f"  Período: {first_date.strftime('%Y-%m')} até {last_date.strftime('%Y-%m')}")

    # 4) Salvar DataFrame consolidado
    out_xlsx = OUTPUT_DIR / "dados_consolidados_macro_credito.xlsx"
    
    base.to_excel(out_xlsx, index=False)

    print("\n" + "="*80)
    print("ARQUIVOS SALVOS")
    print("="*80)
    print(f"XLSX: {out_xlsx}")
    
    # Configurações gerais dos gráficos
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 6.1) Gráfico individual para cada série
    variaveis_plot = [
        ("selic_mensal", "SELIC - Taxa Mensal (%)", "Taxa (% a.m.)"),
        ("ibcbr_dessaz", "IBC-Br Dessazonalizado", "Índice"),
        ("inadimpl_cartao_total", "Inadimplência de Cartão de Crédito", "% do Saldo"),
        ("ipca_mensal", "IPCA - Variação Mensal", "% a.m."),
        ("comprometimento_renda", "Comprometimento de Renda (BCB - até 2021)", "% da Renda"),
        ("endividamento_familias", "Endividamento das Famílias", "%"),
        ("inadimplencia_familias", "Inadimplência das Famílias", "%"),
        ("servico_divida", "Serviço da Dívida das Famílias", "R$ milhões"),
        ("renda_disponivel", "Renda Disponível das Famílias", "R$ milhões"),
    ]
    
    for col, titulo, ylabel in variaveis_plot:
        if col in base.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(base["data"], base[col], linewidth=2, color='steelblue')
            ax.set_title(titulo, fontsize=14, fontweight='bold')
            ax.set_xlabel("Data", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = OUTPUT_DIR / f"grafico_{col}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico salvo: {filename.name}")
            plt.close()
    
    # 6.2) Gráfico comparativo: Variáveis de Pressão Financeira das Famílias
    fig, ax = plt.subplots(figsize=(14, 7))
    
    if "comprometimento_renda" in base.columns:
        ax.plot(base["data"], base["comprometimento_renda"], 
                label="Comprometimento de Renda (até 2021)", linewidth=2.5, color='darkred')
    
    if "endividamento_familias" in base.columns:
        ax.plot(base["data"], base["endividamento_familias"], 
                label="Endividamento", linewidth=2, color='darkorange')
    
    if "inadimplencia_familias" in base.columns:
        ax.plot(base["data"], base["inadimplencia_familias"], 
                label="Inadimplência", linewidth=2, color='purple')
    
    ax.set_title("Indicadores de Pressão Financeira das Famílias", fontsize=14, fontweight='bold')
    ax.set_xlabel("Data", fontsize=12)
    ax.set_ylabel("Percentual (%)", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = OUTPUT_DIR / "grafico_comparativo_familias.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico salvo: {filename.name}")
    plt.close()
    
    # 6.3) Gráfico comparativo: Inadimplência Cartão vs Famílias
    if "inadimpl_cartao_total" in base.columns and "inadimplencia_familias" in base.columns:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(base["data"], base["inadimpl_cartao_total"], 
                label="Inadimplência Cartão de Crédito", linewidth=2.5, color='crimson')
        ax.plot(base["data"], base["inadimplencia_familias"], 
                label="Inadimplência Famílias (Total)", linewidth=2, color='navy', linestyle='--')
        
        ax.set_title("Comparação: Inadimplência de Cartão vs. Inadimplência Total das Famílias", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Data", fontsize=12)
        ax.set_ylabel("Percentual (%)", fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = OUTPUT_DIR / "grafico_inadimplencia_comparacao.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {filename.name}")
        plt.close()
    
    # 6.4) Gráfico comparativo: Serviço da Dívida vs Renda Disponível
    if "servico_divida" in base.columns and "renda_disponivel" in base.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Painel superior: Ambas as séries
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(base["data"], base["servico_divida"], 
                label="Serviço da Dívida", linewidth=2.5, color='darkred')
        line2 = ax1_twin.plot(base["data"], base["renda_disponivel"], 
                label="Renda Disponível", linewidth=2, color='darkgreen', linestyle='--')
        
        ax1.set_title("Serviço da Dívida vs. Renda Disponível das Famílias", 
                    fontsize=14, fontweight='bold')
        ax1.set_xlabel("Data", fontsize=12)
        ax1.set_ylabel("Serviço da Dívida (R$ milhões)", fontsize=11, color='darkred')
        ax1_twin.set_ylabel("Renda Disponível (R$ milhões)", fontsize=11, color='darkgreen')
        ax1.tick_params(axis='y', labelcolor='darkred')
        ax1_twin.tick_params(axis='y', labelcolor='darkgreen')
        
        # Combinar legendas
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Painel inferior: Razão entre elas
        razao = (base["servico_divida"] / base["renda_disponivel"]) * 100
        ax2.plot(base["data"], razao, linewidth=2.5, color='purple')
        ax2.set_title("Razão: Serviço da Dívida / Renda Disponível", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Data", fontsize=12)
        ax2.set_ylabel("Percentual (%)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=100, color='red', linestyle=':', linewidth=1, alpha=0.5, label='100%')
        
        plt.tight_layout()
        
        filename = OUTPUT_DIR / "grafico_servico_divida_renda.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {filename.name}")
        plt.close()
    
    # 6.4) Dashboard com 4 variáveis principais
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Dashboard: Variáveis Macroeconômicas e de Crédito", fontsize=16, fontweight='bold')
    
    # SELIC
    if "selic_mensal" in base.columns:
        axes[0, 0].plot(base["data"], base["selic_mensal"], linewidth=2, color='darkgreen')
        axes[0, 0].set_title("SELIC - Taxa Mensal", fontweight='bold')
        axes[0, 0].set_ylabel("% a.m.")
        axes[0, 0].grid(True, alpha=0.3)
    
    # IBC-Br
    if "ibcbr_dessaz" in base.columns:
        axes[0, 1].plot(base["data"], base["ibcbr_dessaz"], linewidth=2, color='steelblue')
        axes[0, 1].set_title("IBC-Br Dessazonalizado", fontweight='bold')
        axes[0, 1].set_ylabel("Índice")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Inadimplência Cartão
    if "inadimpl_cartao_total" in base.columns:
        axes[1, 0].plot(base["data"], base["inadimpl_cartao_total"], linewidth=2, color='crimson')
        axes[1, 0].set_title("Inadimplência Cartão de Crédito", fontweight='bold')
        axes[1, 0].set_ylabel("% do Saldo")
        axes[1, 0].set_xlabel("Data")
        axes[1, 0].grid(True, alpha=0.3)
    
    # Endividamento Famílias
    if "endividamento_familias" in base.columns:
        axes[1, 1].plot(base["data"], base["endividamento_familias"], linewidth=2, color='darkorange')
        axes[1, 1].set_title("Endividamento das Famílias", fontweight='bold')
        axes[1, 1].set_ylabel("%")
        axes[1, 1].set_xlabel("Data")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = OUTPUT_DIR / "dashboard_principal.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Dashboard salvo: {filename.name}")
    plt.close()
    
    print("\n" + "="*80)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print(f"\nTodos os arquivos foram salvos em:")
    print(f"{OUTPUT_DIR}")
    print("\nArquivos gerados:")
    print("  • dados_consolidados_macro_credito.xlsx")
    print("  • Múltiplos gráficos em formato PNG (alta resolução)")
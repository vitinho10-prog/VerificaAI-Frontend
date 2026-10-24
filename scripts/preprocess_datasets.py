"""
VerificAI - Script de Pré-processamento e Auditoria de Datasets
==============================================================
Este script realiza a auditoria completa de qualidade e o pré-processamento
dos 3 datasets oficiais do projeto localizados na pasta `database/`:
  1. E-mails em Inglês (CEAS_08)
  2. SMS em Inglês (Mendeley Data - Munoz & Islam)
  3. SMS / WhatsApp em Português (SBSeg 2023 - Nascimento et al. / UFC)

Ações executadas:
  - Verificação e remoção de IDs nulos ou geração de IDs únicos sequenciais.
  - Tratamento de campos nulos (ex: subjects vazios em e-mails).
  - Remoção de duplicatas exatas de linha e duplicatas de texto.
  - Resolução de conflitos de rotulagem em mensagens idênticas.
  - Normalização e padronização das colunas ('id', 'text', 'label', 'label_eval', 'label_binary').
  - Criação de datasets limpos completos (`database/cleaned_*.csv`).
  - Criação de amostras balanceadas para avaliação rápida (`database/eval_*_100.csv`).
  - Criação de links/cópias de compatibilidade para o RAG (`database/email_dataset.csv` e `database/sms_dataset.csv`).
"""

import os
import sys
import re
import shutil
import pandas as pd
import numpy as np

# Garante suporte a UTF-8 no stdout do Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATABASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database"))

RAW_FILES = {
    "email_en": os.path.join(DATABASE_DIR, "dataset_email_ingles.csv"),
    "sms_en": os.path.join(DATABASE_DIR, "dataset_sms_ingles.csv"),
    "sms_pt": os.path.join(DATABASE_DIR, "dataset_sms_pt_br.csv"),
}

OUTPUT_CLEAN = {
    "email_en": os.path.join(DATABASE_DIR, "cleaned_email_ingles.csv"),
    "sms_en": os.path.join(DATABASE_DIR, "cleaned_sms_ingles.csv"),
    "sms_pt": os.path.join(DATABASE_DIR, "cleaned_sms_pt_br.csv"),
}

OUTPUT_EVAL = {
    "email_en": os.path.join(DATABASE_DIR, "eval_email_ingles_100.csv"),
    "sms_en": os.path.join(DATABASE_DIR, "eval_sms_ingles_100.csv"),
    "sms_pt": os.path.join(DATABASE_DIR, "eval_sms_pt_br_100.csv"),
}


def normalize_whitespace(text: str) -> str:
    """Remove espacos excessivos e quebras repetitivas mantendo a estrutura basica."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def preprocess_email_en():
    path = RAW_FILES["email_en"]
    print(f"\n{'='*60}")
    print(f"-> Processando: E-mails em Ingles (CEAS_08)")
    print(f"   Arquivo: {path}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"[ERRO] Arquivo nao encontrado: {path}")
        return None

    df = pd.read_csv(path, low_memory=False)
    initial_len = len(df)
    print(f"Total de registros originais: {initial_len:,}")

    # 1. Tratamento de campos de texto
    subject = df["subject"].fillna("").astype(str).str.strip()
    body = df["body"].fillna("").astype(str).str.strip()

    # Formata como e-mail real legivel para o LLM
    text_combined = np.where(
        subject != "",
        "Subject: " + subject + "\n\n" + body,
        body
    )
    df["text"] = [normalize_whitespace(t) for t in text_combined]

    # 2. Filtro de textos vazios ou invalidos
    valid_text_mask = df["text"].str.len() >= 10
    dropped_empty = (~valid_text_mask).sum()
    df = df[valid_text_mask].copy()

    # 3. Tratamento de rotulos (1 = phishing/spam, 0 = safe/ham)
    df["label_binary"] = df["label"].astype(int)
    df["label"] = df["label_binary"].map({1: "phishing", 0: "safe"})
    df["label_eval"] = df["label"]  # Utilizado em scripts de avaliacao binaria

    # 4. Auditoria de duplicatas de texto
    dup_texts = df["text"].duplicated().sum()
    df = df.drop_duplicates(subset=["text"], keep="first").copy()

    # 5. Geracao de IDs unicos e sem valores nulos
    df["id"] = [f"EMAIL_EN_{i+1:05d}" for i in range(len(df))]

    # Selecao e ordenacao de colunas
    cols_order = ["id", "text", "label", "label_eval", "label_binary", "sender", "receiver", "date", "urls"]
    available_cols = [c for c in cols_order if c in df.columns]
    df_clean = df[available_cols].reset_index(drop=True)

    print(f"  - Registros vazios/curtos removidos: {dropped_empty}")
    print(f"  - Textos duplicados removidos: {dup_texts}")
    print(f"  - Total final limpo: {len(df_clean):,}")
    print(f"  - Distribuicao: {dict(df_clean['label'].value_counts())}")

    df_clean.to_csv(OUTPUT_CLEAN["email_en"], index=False, encoding="utf-8")
    print(f"  [OK] Salvo em: {OUTPUT_CLEAN['email_en']}")

    # Amostra balanceada estratificada (50% positivos / 50% negativos)
    create_balanced_eval_split(df_clean, OUTPUT_EVAL["email_en"], sample_size=100)

    return df_clean


def preprocess_sms_en():
    path = RAW_FILES["sms_en"]
    print(f"\n{'='*60}")
    print(f"-> Processando: SMS em Ingles (Mendeley Data)")
    print(f"   Arquivo: {path}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"[ERRO] Arquivo nao encontrado: {path}")
        return None

    df = pd.read_csv(path, low_memory=False)
    initial_len = len(df)
    print(f"Total de registros originais: {initial_len:,}")

    # 1. Normalizacao de texto
    df["text"] = [normalize_whitespace(str(t)) for t in df["TEXT"]]

    # 2. Filtro de textos vazios
    valid_text_mask = df["text"].str.len() >= 5
    dropped_empty = (~valid_text_mask).sum()
    df = df[valid_text_mask].copy()

    # 3. Tratamento de rotulos originais ('ham', 'spam', 'smishing')
    df["label"] = df["LABEL"].astype(str).str.lower().str.strip()
    # Mapeia binario: ham = 0 (safe), spam/smishing = 1 (threat)
    df["label_binary"] = df["label"].map({"ham": 0, "spam": 1, "smishing": 1})
    df["label_eval"] = df["label_binary"].map({1: "spam", 0: "ham"})

    # 4. Remover duplicatas de texto resolvendo conflitos
    # Se houver mesmo texto rotulado diferente, prioriza smishing > spam > ham
    df = df.sort_values(by=["label_binary", "label"], ascending=[False, False])
    dup_texts = df["text"].duplicated().sum()
    df = df.drop_duplicates(subset=["text"], keep="first").copy()

    # 5. Geracao de IDs unicos
    df["id"] = [f"SMS_EN_{i+1:05d}" for i in range(len(df))]

    # Renomear metadados uteis
    if "URL" in df.columns:
        df["has_url"] = df["URL"]
    if "EMAIL" in df.columns:
        df["has_email"] = df["EMAIL"]
    if "PHONE" in df.columns:
        df["has_phone"] = df["PHONE"]

    cols_order = ["id", "text", "label", "label_eval", "label_binary", "has_url", "has_email", "has_phone"]
    available_cols = [c for c in cols_order if c in df.columns]
    df_clean = df[available_cols].reset_index(drop=True)

    print(f"  - Registros vazios/curtos removidos: {dropped_empty}")
    print(f"  - Textos duplicados removidos (artificiais): {dup_texts}")
    print(f"  - Total final limpo: {len(df_clean):,}")
    print(f"  - Distribuicao Categorica: {dict(df_clean['label'].value_counts())}")
    print(f"  - Distribuicao Binaria (0=Safe/Ham, 1=Threat/Spam): {dict(df_clean['label_binary'].value_counts())}")

    df_clean.to_csv(OUTPUT_CLEAN["sms_en"], index=False, encoding="utf-8")
    print(f"  [OK] Salvo em: {OUTPUT_CLEAN['sms_en']}")

    create_balanced_eval_split(df_clean, OUTPUT_EVAL["sms_en"], sample_size=100)

    return df_clean


def preprocess_sms_pt():
    path = RAW_FILES["sms_pt"]
    print(f"\n{'='*60}")
    print(f"-> Processando: SMS / Mensagens PT-BR (SBSeg 2023 - UFC)")
    print(f"   Arquivo: {path}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"[ERRO] Arquivo nao encontrado: {path}")
        return None

    df = pd.read_csv(path, low_memory=False)
    initial_len = len(df)
    print(f"Total de registros originais: {initial_len:,}")

    # 1. Normalizacao de texto (usa text_content bruto original que preserva links e pontuacao)
    df["text"] = [normalize_whitespace(str(t)) for t in df["text_content"]]

    # 2. Filtro de textos validos (minimo de 10 caracteres)
    valid_text_mask = df["text"].str.len() >= 10
    dropped_empty = (~valid_text_mask).sum()
    df = df[valid_text_mask].copy()

    # 3. Tratamento de rotulos (Golpe: 1 = golpe, 0 = legitimo)
    df["label_binary"] = df["Golpe"].astype(int)
    df["label"] = df["label_binary"].map({1: "golpe", 0: "legitimo"})
    df["label_eval"] = df["label"]

    # 4. Remover duplicatas em massa (mensagens identicas replicadas em grupos de Telegram)
    dup_texts = df["text"].duplicated().sum()
    df = df.drop_duplicates(subset=["text"], keep="first").copy()

    # 5. Higienizacao de IDs (garante unicidade e ausencia de nulos)
    if "id" in df.columns and df["id"].notna().all() and not df["id"].duplicated().any():
        df["id"] = df["id"].astype(str)
    else:
        df["id"] = [f"BR_{i+1:05d}" for i in range(len(df))]

    cols_order = ["id", "text", "label", "label_eval", "label_binary", "date_message", "has_media_url"]
    available_cols = [c for c in cols_order if c in df.columns]
    df_clean = df[available_cols].reset_index(drop=True)

    print(f"  - Registros vazios/curtos removidos: {dropped_empty}")
    print(f"  - Mensagens replicadas/duplicadas removidas: {dup_texts:,}")
    print(f"  - Total final de mensagens unicas: {len(df_clean):,}")
    print(f"  - Distribuicao: {dict(df_clean['label'].value_counts())}")

    df_clean.to_csv(OUTPUT_CLEAN["sms_pt"], index=False, encoding="utf-8")
    print(f"  [OK] Salvo em: {OUTPUT_CLEAN['sms_pt']}")

    create_balanced_eval_split(df_clean, OUTPUT_EVAL["sms_pt"], sample_size=100)

    return df_clean


def create_balanced_eval_split(df, output_path, sample_size=100):
    """Cria uma particao de teste balanceada (50% positivos, 50% negativos)."""
    n_per_class = sample_size // 2

    df_pos = df[df["label_binary"] == 1]
    df_neg = df[df["label_binary"] == 0]

    sample_pos = df_pos.sample(n=min(n_per_class, len(df_pos)), random_state=42)
    sample_neg = df_neg.sample(n=min(n_per_class, len(df_neg)), random_state=42)

    eval_df = pd.concat([sample_pos, sample_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    eval_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"  [EVAL] Amostra Balanceada ({len(eval_df)} linhas: 50% Pos / 50% Neg) salva em: {output_path}")


def setup_legacy_rag_files():
    """Garante compatibilidade com servicos antigos criando database/email_dataset.csv e database/sms_dataset.csv."""
    email_clean = OUTPUT_CLEAN["email_en"]
    sms_pt_clean = OUTPUT_CLEAN["sms_pt"]

    legacy_email = os.path.join(DATABASE_DIR, "email_dataset.csv")
    legacy_sms = os.path.join(DATABASE_DIR, "sms_dataset.csv")

    if os.path.exists(email_clean):
        shutil.copyfile(email_clean, legacy_email)
        print(f"  [RAG] Arquivo legado gerado: {legacy_email}")

    if os.path.exists(sms_pt_clean):
        shutil.copyfile(sms_pt_clean, legacy_sms)
        print(f"  [RAG] Arquivo legado gerado: {legacy_sms}")


def main():
    print("\nIniciando Pipeline de Pre-processamento e Auditoria dos Datasets...")
    preprocess_email_en()
    preprocess_sms_en()
    preprocess_sms_pt()
    setup_legacy_rag_files()
    print("\n" + "="*60)
    print("[SUCESSO] Todos os 3 datasets foram auditados, limpos e padronizados com sucesso!")
    print("="*60)


if __name__ == "__main__":
    main()

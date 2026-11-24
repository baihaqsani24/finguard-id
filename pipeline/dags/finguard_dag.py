"""
========================================================
Final Project — FinGuard ID

Anggota Tim (Batch HCK-45):
- Alyssa Putri Azahra
- Baihaqsani
- Rizqullah Anjar Pradana

Skrip ini dikhususkan untuk pipeline otomatisasi ETL dan
batch scoring debitur FinGuard ID dari PostgreSQL ke
Elasticsearch menggunakan Apache Airflow (dijalankan lokal
via pipeline/airflow_ES.yaml).
========================================================
"""

import os
import warnings
warnings.filterwarnings('ignore', message=r'X does not have valid feature names.*LGBMRegressor', category=UserWarning)

import sys
import json
import datetime as dt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from elasticsearch import Elasticsearch, helpers
from airflow import DAG
from airflow.operators.python import PythonOperator

# Pastikan modul features.py di folder dags/ terbaca saat import dan unpickle model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from features import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    FLAG_COLS,
    RAW_INPUT,
    TARGET,
    add_features,
    dpd_to_collectibility,
    dpd_to_eligibility,
    predict_dpd,
    simulate_restructuring,
)

# Endpoint PostgreSQL & Elasticsearch (diambil dari file .env di folder pipeline/)
DB_CONN = os.getenv(
    'AIRFLOW__DATABASE__SQL_ALCHEMY_CONN',
    'postgresql+psycopg2://airflow:airflow@postgres/airflow'
)
ES_CONN = os.getenv(
    'ELASTICSEARCH_HOSTS',
    'http://elasticsearch:9200'
)

# Nama tabel PostgreSQL dan nama index Elasticsearch
TABLE_NAME = 'finguard_raw'
ES_INDEX = 'finguard_debitur'

# Path file sementara di dalam container Airflow (/opt/airflow/dags/)
MODEL_DIR = os.path.join(BASE_DIR, 'model')
RAW_OUTPUT = os.path.join(BASE_DIR, 'finguard_extract.csv')
CLEAN_OUTPUT = os.path.join(BASE_DIR, 'finguard_transformed.csv')
SCORED_OUTPUT = os.path.join(BASE_DIR, 'finguard_scored.csv')


def fetch_psql(db_conn=DB_CONN, table_name=TABLE_NAME, raw_output=RAW_OUTPUT):
    """
    Mengambil seluruh data mentah debitur (15 kolom) dari PostgreSQL
    dan menyimpannya ke file CSV untuk tahap validasi dan transformasi.
    """
    engine = create_engine(db_conn)
    query = f'SELECT * FROM {table_name}'
    df_raw = pd.read_sql(query, con=engine)
    df_raw.to_csv(raw_output, index=False)
    print(f'[fetch_psql] Berhasil menarik {len(df_raw):,} baris dari tabel {table_name}.')
    return raw_output


def validate_data(raw_output=RAW_OUTPUT):
    """
    Memvalidasi kualitas data mentah (aturan utama sebelum transformasi):
    ID unik & tidak null, rentang usia 18-35, DPD 0-180, penghasilan > 0, dll.
    """
    df = pd.read_csv(raw_output)
    assert df['borrower_id'].notna().all() and df['borrower_id'].is_unique, 'borrower_id harus unik dan tidak null'
    assert df['age'].between(18, 35).all(), 'age di luar rentang 18-35'
    assert df['days_past_due'].between(0, 180).all(), 'days_past_due di luar rentang 0-180'
    assert (df['monthly_income_idr'] > 0).all(), 'monthly_income_idr harus > 0'
    assert df['num_active_lending_apps'].between(1, 12).all(), 'num_active_lending_apps di luar 1-12'
    assert df['daily_interest_rate_pct'].between(0.0, 1.0).all(), 'daily_interest_rate_pct di luar 0-1'
    assert df['loan_tenor_months'].isin([1, 3, 6, 9, 12, 18, 24]).all(), 'loan_tenor_months tidak valid'
    print(f'[validate_data] Validasi {len(df):,} baris lolos semua aturan.')
    return raw_output


def transform_features(raw_output=RAW_OUTPUT, clean_output=CLEAN_OUTPUT):
    """
    Menghitung fitur rasio turunan dan 4 flag kepatuhan OJK menggunakan
    modul bersama features.py (tanpa fit statistik agar bebas leakage).
    """
    df = pd.read_csv(raw_output)
    df_feat = add_features(df)
    df_feat['n_pelanggaran'] = df_feat[FLAG_COLS].sum(axis=1)
    df_feat['is_macet'] = (df_feat[TARGET] > 90).astype(int)
    df_feat['kolektibilitas_aktual'] = dpd_to_collectibility(df_feat[TARGET], DEFAULT_THRESHOLDS)
    df_feat['dsr_bucket'] = pd.cut(
        df_feat['debt_service_ratio'],
        bins=[0.0, 0.30, 1.00, 2.00, 100.0],
        labels=['<= 0.3', '0.3 - 1.0', '1.0 - 2.0', '> 2.0']
    ).astype(str)
    df_feat['snapshot_date'] = dt.date.today().isoformat()
    df_feat.to_csv(clean_output, index=False)
    print(f'[transform_features] Data hasil transformasi disimpan ke {clean_output}.')
    return clean_output


def score_model(clean_output=CLEAN_OUTPUT, scored_output=SCORED_OUTPUT, model_dir=MODEL_DIR):
    """
    Melakukan batch scoring dengan Model 1 (model_kelayakan.pkl) dan simulasi
    restrukturisasi Model 2 (model_restrukturisasi.pkl) untuk debitur DPD 31-90 hari.
    Jika library ML (scikit-learn/lightgbm) belum terpasang di container Airflow,
    pipeline tetap berjalan pada Level 1 (tanpa skor model) agar tidak gagal.
    """
    df = pd.read_csv(clean_output)
    m1_path = os.path.join(model_dir, 'model_kelayakan.pkl')
    m2_path = os.path.join(model_dir, 'model_restrukturisasi.pkl')
    meta_path = os.path.join(model_dir, 'model_metadata.json')
    test_ids_path = os.path.join(model_dir, 'test_ids.csv')

    thr = DEFAULT_THRESHOLDS
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            thr = json.load(f).get('thresholds', DEFAULT_THRESHOLDS)

    try:
        import joblib
        m1 = joblib.load(m1_path)
        m2 = joblib.load(m2_path)

        preds_dpd = predict_dpd(m1, df[RAW_INPUT])
        df['pred_dpd'] = np.round(preds_dpd, 2)
        df['pred_kolektibilitas'] = dpd_to_collectibility(preds_dpd, thr)
        df['pred_kelayakan'] = dpd_to_eligibility(preds_dpd, thr)

        # Jalankan simulasi Model 2 untuk debitur prediksi DPD 31-90 hari
        tenor_rec = pd.Series(np.nan, index=df.index)
        dpd_after = pd.Series(np.nan, index=df.index)
        perlu_lain = pd.Series(None, index=df.index, dtype=object)

        # filter memakai prediksi SEBELUM dibulatkan (konsisten dengan pred_kelayakan)
        mask_mid = (preds_dpd > 30.0) & (preds_dpd <= 90.0)
        for idx in df.index[mask_mid]:
            sim = simulate_restructuring(df.loc[[idx], RAW_INPUT], m2, thr=thr)
            if sim['best'] is not None:                    # BUKAN 'recommended'
                tenor_rec.loc[idx] = int(sim['best']['tenor_bulan'])
                dpd_after.loc[idx] = float(sim['best']['pred_dpd_after'])
                perlu_lain.loc[idx] = 'tidak'
            else:                                          # BUKAN 'haircut_scenario'
                tenor_rec.loc[idx] = 24
                dpd_after.loc[idx] = float(sim['haircut']['pred_dpd_after'])
                perlu_lain.loc[idx] = 'ya'

        df['tenor_rekomendasi'] = tenor_rec
        df['pred_dpd_after'] = np.round(dpd_after, 2)
        df['perlu_penanganan_lain'] = perlu_lain
        df['scoring_mode'] = 'level2-dual-regression'
        print('[score_model] Batch scoring Level 2 (Model 1 + Model 2) selesai.')
    except Exception as e:
        # Jangan pernah mengisi kolom prediksi dengan nilai aktual.
        # Kolom dikosongkan agar dashboard tidak menampilkan hasil palsu.
        print(f'[score_model] GAGAL scoring ML -> Level 1 (kolom prediksi dikosongkan). Penyebab: {e!r}')
        for c in ['pred_dpd', 'tenor_rekomendasi', 'pred_dpd_after']:
            df[c] = np.nan
        for c in ['pred_kolektibilitas', 'pred_kelayakan', 'perlu_penanganan_lain']:
            df[c] = None
        df['scoring_mode'] = 'level1-etl-only'

    # Tandai split train/test berdasarkan test_ids.csv untuk filter evaluasi di Kibana
    if os.path.exists(test_ids_path):
        test_set = set(pd.read_csv(test_ids_path)['borrower_id'].tolist())
        df['split'] = np.where(df['borrower_id'].isin(test_set), 'test', 'train')
    else:
        df['split'] = 'all'

    df.to_csv(scored_output, index=False)
    return scored_output


def insert_to_elastic(scored_input=SCORED_OUTPUT, es_conn=ES_CONN, index_name=ES_INDEX):
    """
    Memuat data hasil transformasi & scoring ke Elasticsearch menggunakan
    bulk indexing dengan borrower_id sebagai _id agar bersifat idempotent.
    """
    es = Elasticsearch(es_conn)
    df_elastic = pd.read_csv(scored_input)

    actions = [
        {
            '_index': index_name,
            '_id': str(row['borrower_id']),
            '_source': row.to_dict()
        }
        for _, row in df_elastic.iterrows()
    ]

    success_count, _ = helpers.bulk(es, actions)
    print(f'[insert_to_elastic] Berhasil memuat {success_count:,} dokumen ke index {index_name}.')
    return success_count


default_args = {
    'owner': 'FinGuard_HCK45',
    'start_date': dt.datetime(2026, 11, 1),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=2),
}

with DAG(
    dag_id='FinGuard_ID_DAG',
    default_args=default_args,
    schedule_interval='0 7 * * *',
    catchup=False,
    description='ETL & Batch Scoring Pipeline FinGuard ID dari PostgreSQL ke Elasticsearch'
) as dag:

    fetch = PythonOperator(
        task_id='fetch_psql',
        python_callable=fetch_psql
    )

    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data
    )

    transform = PythonOperator(
        task_id='transform_features',
        python_callable=transform_features
    )

    score = PythonOperator(
        task_id='score_model',
        python_callable=score_model
    )

    insert = PythonOperator(
        task_id='insert_to_elastic',
        python_callable=insert_to_elastic
    )

    # Urutan Task
    fetch >> validate >> transform >> score >> insert

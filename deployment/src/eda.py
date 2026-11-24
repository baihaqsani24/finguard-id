"""
========================================================
Final Project — FinGuard ID (Batch HCK-45)

Halaman Exploratory Data Analysis (EDA), Audit VIF,
dan Kepatuhan 4 Aturan OJK berbasis Days Past Due (DPD).
========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from features import (
    COLLECT_ORDER,
    DEFAULT_THRESHOLDS,
    FLAG_COLS,
    MODEL_NUM_COLS,
    TARGET,
    add_features,
    dpd_to_collectibility,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data():
    """Memuat data mentah 15 kolom dan menghitung fitur turunan lewat features.py."""
    data_path = os.path.join(BASE_DIR, 'finguard_raw.csv')
    df = pd.read_csv(data_path)
    df = add_features(df)
    df['kolektibilitas'] = dpd_to_collectibility(df[TARGET], DEFAULT_THRESHOLDS)
    df['n_pelanggaran'] = df[FLAG_COLS].sum(axis=1)
    return df


def hitung_vif(df, cols):
    """Menghitung Variance Inflation Factor (VIF) untuk memastikan seluruh fitur < 10."""
    X_mat = df[cols].astype(float).values
    X_const = np.column_stack([np.ones(len(X_mat)), X_mat])
    rows = []
    for i, col in enumerate(cols):
        y_col = X_mat[:, i]
        X_other = np.delete(X_const, i + 1, axis=1)
        coef, _, _, _ = np.linalg.lstsq(X_other, y_col, rcond=None)
        ss_res = np.sum((y_col - X_other @ coef) ** 2)
        ss_tot = np.sum((y_col - np.mean(y_col)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        vif_val = np.inf if (1.0 - r2) < 1e-12 else 1.0 / (1.0 - r2)
        rows.append({'Fitur': col, 'VIF': round(float(vif_val), 2)})
    return pd.DataFrame(rows).sort_values('VIF', ascending=False).reset_index(drop=True)


def run():
    """Fungsi untuk menjalankan dashboard EDA FinGuard ID."""
    logo_path = os.path.join(BASE_DIR, 'finguard_logo.png')
    if os.path.exists(logo_path):
        st.image(logo_path, width=360)

    st.title('Exploratory Data Analysis — FinGuard ID')
    st.subheader('Analisis Profil Debitur, Days Past Due (DPD), Audit VIF & Kepatuhan OJK')

    df = load_data()

    # Ringkasan metrik utama portofolio
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Debitur', f'{len(df):,}')
    c2.metric('Rata-rata DPD', f"{df[TARGET].mean():.1f} hari")
    c3.metric('Menunggak (DPD > 0)', f"{int((df[TARGET] > 0).sum()):,} ({(df[TARGET] > 0).mean()*100:.1f}%)")
    c4.metric('Macet (DPD > 90)', f"{int((df[TARGET] > 90).sum()):,} ({(df[TARGET] > 90).mean()*100:.1f}%)")

    st.write('### Sampel Data Debitur (15 Kolom Mentah + Fitur Turunan)')
    st.dataframe(df.head(10), use_container_width=True)

    # Visualisasi 1: Distribusi Days Past Due & 5 Rentang Kolektibilitas
    st.write('### 1. Distribusi Days Past Due (DPD) & 5 Kelas Kolektibilitas OJK')
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plt.figure(figsize=(8, 4))
        sns.histplot(df[TARGET], bins=30, color='#1b365d', edgecolor='white')
        plt.axvline(30, color='#e08244', linestyle='--', label='Batas 30 hari (DPK)')
        plt.axvline(90, color='#c2591b', linestyle='--', label='Batas 90 hari (Macet / TWP90)')
        plt.xlabel('Days Past Due (Hari)')
        plt.ylabel('Jumlah Debitur')
        plt.title('Distribusi Kontinu DPD (0–180 Hari)')
        plt.legend()
        st.pyplot(fig1)

    with col2:
        fig2 = plt.figure(figsize=(8, 4))
        cnt = df['kolektibilitas'].value_counts().reindex(COLLECT_ORDER)
        sns.barplot(x=cnt.values, y=cnt.index, palette=['#1b365d', '#3b6998', '#6b9ac4', '#e08244', '#c2591b'])
        plt.xlabel('Jumlah Debitur')
        plt.ylabel('Kolektibilitas OJK')
        plt.title('Jumlah Debitur per Kelas Kolektibilitas')
        st.pyplot(fig2)

    # Visualisasi 2: Rata-rata DPD per Tujuan Pinjaman & Rentang DSR
    st.write('### 2. Rata-rata DPD Berdasarkan Tujuan Pinjaman & Rentang DSR')
    col3, col4 = st.columns(2)
    with col3:
        fig3 = plt.figure(figsize=(8, 4.2))
        purp_dpd = df.groupby('primary_loan_purpose')[TARGET].mean().sort_values()
        plt.barh(purp_dpd.index, purp_dpd.values, color='#c2591b')
        plt.xlabel('Rata-rata DPD (Hari)')
        plt.title('Rata-rata DPD per Tujuan Pinjaman')
        st.pyplot(fig3)

    with col4:
        fig4 = plt.figure(figsize=(8, 4.2))
        dsr_bins = pd.cut(
            df['debt_service_ratio'],
            bins=[0.0, 0.30, 0.60, 1.00, 10.0],
            labels=['<= 30% (OJK)', '30%–60%', '60%–100%', '> 100%']
        )
        dsr_dpd = df.groupby(dsr_bins, observed=False)[TARGET].mean()
        plt.bar(dsr_dpd.index.astype(str), dsr_dpd.values, color=['#1b365d', '#6b9ac4', '#e08244', '#c2591b'])
        plt.ylabel('Rata-rata DPD (Hari)')
        plt.xlabel('Rentang Debt Service Ratio (DSR)')
        plt.title('Lonjakan DPD Berdasarkan Rentang DSR (Batas OJK 30%)')
        st.pyplot(fig4)

    # Visualisasi 3: Kepatuhan 4 Aturan OJK & Tabel Audit VIF
    st.write('### 3. Kepatuhan 4 Aturan OJK & Audit Multikolinearitas (VIF < 10)')
    col5, col6 = st.columns(2)
    with col5:
        st.write('**Rata-rata DPD per Jumlah Pelanggaran OJK (0–4 Flag):**')
        pel_df = df.groupby('n_pelanggaran').agg(
            jumlah_debitur=('borrower_id', 'count'),
            rata_rata_dpd=(TARGET, lambda x: round(x.mean(), 1)),
            persen_macet=(TARGET, lambda x: round((x > 90).mean() * 100, 1))
        ).reset_index()
        st.dataframe(pel_df, hide_index=True, use_container_width=True)

    with col6:
        st.write('**Hasil Audit VIF Fitur Prediktor Final (`MODEL_NUM_COLS`):**')
        st.dataframe(hitung_vif(df, MODEL_NUM_COLS), hide_index=True, use_container_width=True)

    # Plot Interaktif (gaya P1-M2)
    st.write('### 4. Eksplorasi Fitur Numerik (Interaktif)')
    opsi = st.selectbox(
        'Pilih Kolom Numerik:',
        ('days_past_due', 'debt_service_ratio', 'monthly_income_idr', 'total_loan_outstanding_idr', 'num_active_lending_apps', 'age')
    )
    fig5 = plt.figure(figsize=(10, 4))
    sns.histplot(df[opsi], bins=30, kde=True, color='#1b365d')
    plt.title(f'Distribusi Kolom {opsi}')
    st.pyplot(fig5)


if __name__ == '__main__':
    run()

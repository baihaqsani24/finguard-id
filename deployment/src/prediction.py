"""
========================================================
Final Project — FinGuard ID (Batch HCK-45)

Halaman Prediksi Kelayakan (Model 1: model_kelayakan.pkl),
Simulasi Restrukturisasi Before vs After (Model 2:
model_restrukturisasi.pkl), dan Batch Scoring CSV.
========================================================
"""

import json
import warnings
warnings.filterwarnings('ignore', message=r'X does not have valid feature names.*LGBMRegressor', category=UserWarning)

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import (DSR_CAP, EDU_ORDER, FLAG_COLS, FLAG_INFO, RAW_CAT, RAW_INPUT, SIM_DISCLAIMER, SIM_NO_OPTION,
                      TENOR_OPTIONS, add_features, before_after_table, dpd_to_collectibility, dpd_to_eligibility,
                      ojk_rate_cap, predict_dpd, recommend, simulate_restructuring, top_factors_from_shap)

HERE = os.path.dirname(os.path.abspath(__file__))
STATUS_ICON = {'Layak': '🟢', 'Dalam Pengawasan': '🟠', 'Tidak Layak': '🔴'}


@st.cache_resource
def load_artifacts():
    files = {'m1': 'model_kelayakan.pkl', 'm2': 'model_restrukturisasi.pkl', 'meta': 'model_metadata.json',
             'umr': 'umr_reference.json'}
    paths = {k: os.path.join(HERE, f) for k, f in files.items()}
    missing = [f for k, f in files.items() if not os.path.exists(paths[k])]
    if missing:
        return None, missing
    art = {'m1': joblib.load(paths['m1']), 'm2': joblib.load(paths['m2'])}
    with open(paths['meta']) as f:
        art['meta'] = json.load(f)
    with open(paths['umr']) as f:
        art['umr'] = json.load(f)
    try:
        import shap
        art['explainer'] = shap.TreeExplainer(art['m1']['reg'].regressor_)
    except Exception:  # model non-pohon -> faktor SHAP tidak ditampilkan
        art['explainer'] = None
    return art, []


def get_artifacts():
    art, missing = load_artifacts()
    if missing:
        st.error(f'File model tidak ditemukan: {missing}. Jalankan modeling.ipynb lalu `python sync_artifacts.py`.')
        st.stop()
    return art


def shap_factors(art, row):
    if art['explainer'] is None:
        return []
    Xt = art['m1'][:-1].transform(row[RAW_INPUT])
    Xt = Xt.toarray() if hasattr(Xt, 'toarray') else Xt
    sv = art['explainer'].shap_values(Xt)
    return top_factors_from_shap(sv[0], art['meta']['feature_names_out'], row, top=3)


def check_range(row, meta):
    out = []
    for c, r in meta['input_range'].items():
        v = float(row[c].iloc[0])
        if v < r['min'] or v > r['max']:
            out.append(f"`{c}` = {v:,.2f} di luar rentang data latih ({r['min']:,.2f} – {r['max']:,.2f})")
    return out


def form_input(meta, umr):
    with st.form('form_debitur'):
        st.subheader('Data debitur (input mentah)')
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input('Usia', 17, 70, 24)
            province = st.selectbox('Provinsi (UMR terisi otomatis)', sorted(umr))
            education = st.selectbox('Pendidikan', EDU_ORDER)
            employment = st.selectbox('Pekerjaan', meta['categories']['employment_status'])
        with c2:
            income = st.number_input('Penghasilan bulanan (Rp)', 0, 100_000_000, 5_000_000, step=250_000)
            apps = st.number_input('Jumlah aplikasi pinjol/paylater aktif', 1, 20, 4)
            purpose = st.selectbox('Tujuan pinjaman utama', meta['categories']['primary_loan_purpose'])
        with c3:
            outstanding = st.number_input('Total pokok utang berjalan (Rp)', 0, 200_000_000, 6_000_000, step=250_000)
            limit = st.number_input('Total limit kredit (Rp)', 1, 300_000_000, 9_000_000, step=250_000)
            tenor = st.selectbox('Tenor (bulan)', TENOR_OPTIONS, index=0)
            rate = st.number_input('Bunga harian (%)', 0.0, 1.0, 0.30, step=0.01, format='%.2f')
        ok = st.form_submit_button('Prediksi', type='primary')
    row = pd.DataFrame([{'age': age, 'monthly_income_idr': income, 'regional_minimum_wage_umr_idr': umr[province],
                         'num_active_lending_apps': apps, 'total_loan_outstanding_idr': outstanding,
                         'credit_limit_granted_idr': limit, 'loan_tenor_months': tenor, 'daily_interest_rate_pct': rate,
                         'education_level': education, 'employment_status': employment, 'primary_loan_purpose': purpose}])
    return ok, row


def manual_after(row, m2, tenor, thr):
    """Kolom After untuk tenor pilihan user (bunga mengikuti batas OJK)."""
    sim = row[RAW_INPUT].copy()
    sim['loan_tenor_months'] = tenor
    sim['daily_interest_rate_pct'] = min(float(row['daily_interest_rate_pct'].iloc[0]), float(ojk_rate_cap(tenor)))
    f = add_features(sim).iloc[0]
    d = float(predict_dpd(m2, sim)[0])
    return {'Tenor (bulan)': tenor, 'Bunga harian (%)': float(sim['daily_interest_rate_pct'].iloc[0]),
            'Pokok (Rp)': round(float(f['total_loan_outstanding_idr'])), 'Cicilan bulanan (Rp)': round(float(f['monthly_installment_estimate_idr'])),
            'DSR': f"{f['debt_service_ratio']:.1%}", 'Prediksi DPD (hari)': round(d, 1),
            'Kolektibilitas': dpd_to_collectibility([d], thr)[0], 'Status': dpd_to_eligibility([d], thr)[0]}


def run_single():
    art = get_artifacts(); meta, thr = art['meta'], art['meta']['thresholds']
    st.title('Prediksi DPD & Simulasi Restrukturisasi')
    st.caption(f"Model 1: {meta['model_1']['algoritma']} (test MAE {meta['model_1']['test']['MAE']:.2f} hari) · "
               f"Model 2: {meta['model_2']['algoritma']} (test MAE {meta['model_2']['test']['MAE']:.2f} hari) · batas Lancar {thr['lancar']} hari")
    ok, row = form_input(meta, art['umr'])
    if not (ok or st.session_state.get('sudah_prediksi')):
        return
    st.session_state['sudah_prediksi'] = True
    if row['monthly_income_idr'].iloc[0] <= 0 or row['credit_limit_granted_idr'].iloc[0] <= 0:
        st.error('Penghasilan dan limit kredit harus > 0.'); return
    for w in check_range(row, meta):
        st.warning('⚠️ ' + w + ' — prediksi kurang dapat dipercaya.')

    # ---------------- BEFORE (Model 1)
    f = add_features(row).iloc[0]
    dpd = float(predict_dpd(art['m1'], row)[0])
    kelas, status = dpd_to_collectibility([dpd], thr)[0], dpd_to_eligibility([dpd], thr)[0]
    st.subheader('Before — kondisi sekarang (Model 1)')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Prediksi DPD', f'{dpd:.1f} hari')
    c2.metric('Kolektibilitas', kelas)
    c3.metric('Status kelayakan', f'{STATUS_ICON[status]} {status}')
    c4.metric('DSR', f"{f['debt_service_ratio']:.1%}", delta='> batas 30%' if f['debt_service_ratio'] > DSR_CAP else 'sesuai OJK',
              delta_color='inverse' if f['debt_service_ratio'] > DSR_CAP else 'normal')
    st.success(f"**Rekomendasi:** {recommend(kelas, f['flag_dsr_above_30'])}")
    left, right = st.columns(2)
    with left:
        st.markdown('**3 faktor yang paling menaikkan prediksi DPD (SHAP, dalam hari)**')
        fac = shap_factors(art, row)
        if fac:
            st.dataframe(pd.DataFrame(fac)[['label', 'nilai', 'shap_hari']], hide_index=True, use_container_width=True)
        st.caption(f"Cicilan bulanan Rp{f['monthly_installment_estimate_idr']:,.0f} · DTI {f['debt_to_income_ratio']:.2f}× · "
                   f"utilisasi limit {f['credit_limit_utilization_ratio']:.0%}")
    with right:
        st.markdown('**Label kepatuhan aturan OJK**')
        st.dataframe(pd.DataFrame({'Aturan': [FLAG_INFO[c][0] for c in FLAG_COLS],
                                   'Status': ['❌ Tidak sesuai' if f[c] == 1 else '✅ Sesuai' for c in FLAG_COLS],
                                   'Dasar': [FLAG_INFO[c][1] for c in FLAG_COLS]}), hide_index=True, use_container_width=True)

    # ---------------- AFTER (Model 2)
    st.markdown('---'); st.subheader('After — simulasi restrukturisasi (Model 2)')
    if not (30 < dpd <= 90):
        st.info('Simulasi restrukturisasi hanya untuk prediksi DPD 31–90 hari (Kurang Lancar & Diragukan). '
                + ('Debitur ini cukup ditangani sesuai rekomendasi di atas.' if dpd <= 30 else 'Debitur Macet → collection intensif.'))
        return
    sim = simulate_restructuring(row, art['m2'], thr)
    if sim['options'].empty:
        st.warning('Tenor sudah 24 bulan; tidak ada opsi perpanjangan.')
    else:
        st.markdown('**Opsi tenor** (bunga = min(bunga awal, batas OJK); opsi disarankan = pertama dengan DSR ≤ 30% & DPD after ≤ 30)')
        st.dataframe(sim['options'].style.format({'cicilan_bulanan_idr': 'Rp{:,.0f}', 'dsr': '{:.1%}', 'bunga_harian_pct': '{:.2f}%',
                                                  'pred_dpd_after': '{:.1f}'}), hide_index=True, use_container_width=True)
    if sim['best'] is not None:
        b = sim['best']
        st.success(f"**Opsi disarankan:** tenor {int(b['tenor_bulan'])} bulan · cicilan Rp{b['cicilan_bulanan_idr']:,.0f} · "
                   f"DSR {b['dsr']:.1%} · prediksi DPD after {b['pred_dpd_after']:.1f} hari ({b['kolektibilitas_after']})")
    else:
        h = sim['haircut']
        st.error(SIM_NO_OPTION)
        st.markdown(f"**Informasi keringanan pokok (rumus OJK):** pokok maksimum agar DSR = 30% pada tenor 24 bulan = "
                    f"**Rp{h['pokok_maks_ojk_idr']:,.0f}** (pokok sekarang Rp{h['pokok_sekarang_idr']:,.0f}; kelebihan "
                    f"Rp{h['kelebihan_pokok_idr']:,.0f} / {h['potongan_pct']:.1f}%). Skenario tenor 24 + pokok disesuaikan → "
                    f"prediksi DPD {h['pred_dpd_after']:.1f} hari ({h['kolektibilitas_after']}).")
    st.markdown('**Tabel Before vs After**')
    tab = before_after_table(row, art['m1'], sim, thr)
    if len(sim['options']):
        pilihan = st.selectbox('Bandingkan dengan tenor lain (opsional)', ['—'] + sim['options']['tenor_bulan'].tolist())
        if pilihan != '—':
            tab[f'After (tenor {pilihan} bulan)'] = pd.Series(manual_after(row, art['m2'], int(pilihan), thr))
    st.dataframe(tab.astype(str), use_container_width=True)
    st.caption('⚠️ ' + SIM_DISCLAIMER + ' Kolom "After" adalah prediksi Model 2.')


def score_batch(art, X):
    thr = art['meta']['thresholds']
    feat = add_features(X)
    pred = predict_dpd(art['m1'], X)
    out = X.copy()
    out['pred_dpd'] = pred.round(1)
    out['kolektibilitas'] = dpd_to_collectibility(pred, thr)
    out['status_kelayakan'] = dpd_to_eligibility(pred, thr)
    for c in FLAG_COLS:
        out[c] = feat[c].values
    out['rekomendasi'] = [recommend(k, f) for k, f in zip(out['kolektibilitas'], out['flag_dsr_above_30'])]
    for c in ['tenor_rekomendasi', 'pred_dpd_after', 'dsr_after_pct', 'pokok_maks_ojk_idr']:
        out[c] = np.nan
    out['perlu_penanganan_lain'] = ''
    for i in out.index[(pred > 30) & (pred <= 90)]:
        s = simulate_restructuring(X.loc[[i]], art['m2'], thr)
        if s['best'] is not None:
            out.loc[i, ['tenor_rekomendasi', 'pred_dpd_after', 'dsr_after_pct']] = [s['best']['tenor_bulan'], s['best']['pred_dpd_after'],
                                                                                   round(s['best']['dsr'] * 100, 1)]
            out.loc[i, 'perlu_penanganan_lain'] = 'tidak'
        else:
            h = s['haircut']
            out.loc[i, ['pred_dpd_after', 'dsr_after_pct', 'pokok_maks_ojk_idr']] = [h['pred_dpd_after'], round(h['dsr'] * 100, 1),
                                                                                     h['pokok_maks_ojk_idr']]
            out.loc[i, 'perlu_penanganan_lain'] = 'ya'
    return out.sort_values('pred_dpd', ascending=False)


def run_batch():
    art = get_artifacts(); meta = art['meta']
    st.title('Batch CSV — daftar prioritas collection')
    st.markdown(f"Kolom wajib: `{', '.join(RAW_INPUT)}`. Kolom `regional_minimum_wage_umr_idr` boleh diganti `province` "
                "(UMR diisi otomatis). Kolom tambahan seperti `borrower_id` ikut ditampilkan.")
    up = st.file_uploader('Unggah CSV debitur', type='csv')
    if up is None:
        return
    X = pd.read_csv(up)
    if 'regional_minimum_wage_umr_idr' not in X.columns and 'province' in X.columns:
        X['regional_minimum_wage_umr_idr'] = X['province'].map(art['umr'])
    missing = [c for c in RAW_INPUT if c not in X.columns]
    if missing:
        st.error(f'Kolom tidak lengkap: {missing}'); return
    bad = X[RAW_INPUT].isna().any(axis=1) | (X['monthly_income_idr'] <= 0) | (X['credit_limit_granted_idr'] <= 0)
    if bad.any():
        st.warning(f'{int(bad.sum())} baris dilewati (nilai kosong, provinsi tak dikenal, penghasilan/limit ≤ 0).')
        X = X[~bad].reset_index(drop=True)
    unknown = {c: sorted(set(X[c]) - set(meta['categories'][c])) for c in RAW_CAT}
    unknown = {k: v for k, v in unknown.items() if v}
    if unknown:
        st.warning(f'Kategori tidak dikenal model (diperlakukan sebagai "lainnya"): {unknown}')
    with st.spinner('Memprediksi & menyimulasikan restrukturisasi...'):
        out = score_batch(art, X)
    c1, c2, c3 = st.columns(3)
    for col, s in zip([c1, c2, c3], ['Tidak Layak', 'Dalam Pengawasan', 'Layak']):
        col.metric(s, f"{(out['status_kelayakan'] == s).sum():,}")
    st.dataframe(out, hide_index=True, use_container_width=True)
    st.download_button('Unduh daftar prioritas (CSV)', out.to_csv(index=False).encode('utf-8'), 'finguard_daftar_prioritas.csv', 'text/csv')
    st.caption('⚠️ ' + SIM_DISCLAIMER)

"""
features.py — Modul bersama FinGuard ID (Revisi 4: Dual Regression)
====================================================================

Satu sumber rumus untuk modeling.ipynb, inference.ipynb, aplikasi Streamlit
(deploy/src/features.py) dan DAG Airflow (pipeline/dags/features.py).
Ketiga salinan HARUS identik -> jalankan `python sync_artifacts.py` setelah mengubah file ini.

Kenapa modul terpisah?
    joblib menyimpan *referensi* ke fungsi (nama modul + nama fungsi), bukan kodenya.
    Fungsi yang didefinisikan di notebook (`__main__`) gagal dimuat di Streamlit/Airflow
    (AttributeError: Can't get attribute ...).
"""
import warnings
import numpy as np

warnings.filterwarnings('ignore', message=r'X does not have valid feature names.*LGBMRegressor', category=UserWarning)
import pandas as pd

# ---------------------------------------------------------------------------
# Kolom
# ---------------------------------------------------------------------------
RAW_NUM = ['age', 'monthly_income_idr', 'regional_minimum_wage_umr_idr',
           'num_active_lending_apps', 'total_loan_outstanding_idr',
           'credit_limit_granted_idr', 'loan_tenor_months', 'daily_interest_rate_pct']
RAW_CAT = ['education_level', 'employment_status', 'primary_loan_purpose']
RAW_INPUT = RAW_NUM + RAW_CAT          # 11 kolom mentah = kontrak input model
SENSITIVE = ['gender', 'province']     # tidak masuk model, dipakai uji keadilan
TARGET = 'days_past_due'

# 4 kolom hasil pipeline LLM pengunggah -> dibuang seluruhnya (sesuai proposal)
GENERATIVE_LLM_COLS = ['twp90_default_status', 'primary_default_root_cause',
                       'cot_financial_psychology_rationale', 'confidence_score']

EDU_ORDER = ['High School / SMK', 'Diploma (D3)', 'Bachelor (S1)', 'Master (S2)']
TENOR_OPTIONS = [1, 3, 6, 9, 12, 18, 24]
DPD_MAX = 180

# ---------------------------------------------------------------------------
# Konstanta aturan OJK
# ---------------------------------------------------------------------------
DSR_CAP = 0.30          # SEOJK 19/SEOJK.06/2025: batas rasio cicilan pindar konsumtif 2026
INCOME_MIN = 3_000_000  # penghasilan minimum penerima dana pindar (SP OJK 214) & debitur paylater (PADK 2/2026)
APPS_MAX = 3            # batas platform paylater (PADK 2/2026); indikator risiko untuk pindar

FLAG_COLS = ['flag_rate_above_ojk_cap', 'flag_income_below_3jt',
             'flag_apps_above_3', 'flag_dsr_above_30']
FLAG_INFO = {
    'flag_rate_above_ojk_cap': ('Bunga harian di atas batas OJK (0,3%/hari tenor <= 6 bln; 0,2%/hari tenor > 6 bln)',
                                'SEOJK 19/2023; Siaran Pers OJK SP 214 — pindar konsumtif'),
    'flag_income_below_3jt': ('Penghasilan di bawah Rp3 juta/bulan',
                              'SP OJK 214 (pindar); PADK 2/2026 (paylater multifinance)'),
    'flag_apps_above_3': ('Indikator risiko multi-platform: > 3 aplikasi aktif',
                          'PADK 2/2026 (paylater maks. 3 perusahaan). Batas 3 penyelenggara pindar dicabut Sept 2026'),
    'flag_dsr_above_30': ('Cicilan bulanan > 30% penghasilan (DSR)',
                          'SEOJK 19/SEOJK.06/2025 — 30% untuk pindar konsumtif mulai 2026'),
}

# Fitur model (lolos VIF < 10; lihat modeling.ipynb Bab 5) — dipakai Model 1 & Model 2
MODEL_NUM_COLS = RAW_NUM + ['debt_service_ratio'] + FLAG_COLS
MODEL_CAT_COLS = RAW_CAT

# ---------------------------------------------------------------------------
# Pemetaan DPD -> kolektibilitas & status kelayakan
# ---------------------------------------------------------------------------
# Batas default (= definisi OJK). Batas final hasil penyetelan CV disimpan di model_metadata.json
DEFAULT_THRESHOLDS = {'lancar': 0.5, 'dpk': 30, 'kurang_lancar': 60, 'diragukan': 90}
COLLECT_ORDER = ['Lancar', 'DPK', 'Kurang Lancar', 'Diragukan', 'Macet']
ELIG_ORDER = ['Layak', 'Dalam Pengawasan', 'Tidak Layak']
ACTION_MAP = {'Lancar': 'Pemantauan rutin',
              'DPK': 'Pengingat otomatis',
              'Kurang Lancar': 'Tawaran restrukturisasi (simulasi Model 2)',
              'Diragukan': 'Tawaran restrukturisasi (simulasi Model 2) + kontak negosiasi langsung',
              'Macet': 'Collection intensif; tidak layak tambah pinjaman'}


def dpd_to_collectibility(dpd, thr=DEFAULT_THRESHOLDS):
    """DPD (hari) -> 5 kelas kolektibilitas. Interval (kiri, kanan]: <=lancar | <=30 | <=60 | <=90 | >90."""
    dpd = np.asarray(dpd, dtype=float)
    bins = [-np.inf, thr['lancar'], thr['dpk'], thr['kurang_lancar'], thr['diragukan'], np.inf]
    return np.asarray(pd.cut(dpd, bins=bins, labels=COLLECT_ORDER).astype(str))


def dpd_to_eligibility(dpd, thr=DEFAULT_THRESHOLDS):
    """Layak <= 30 hari | Dalam Pengawasan 31–90 | Tidak Layak > 90."""
    dpd = np.asarray(dpd, dtype=float)
    return np.select([dpd <= thr['dpk'], dpd <= thr['diragukan']],
                     ['Layak', 'Dalam Pengawasan'], default='Tidak Layak')


# ---------------------------------------------------------------------------
# Rumus
# ---------------------------------------------------------------------------
def ojk_rate_cap(tenor_months):
    """Batas bunga harian konsumtif OJK (%): 0,3 untuk tenor <= 6 bulan, 0,2 untuk > 6 bulan."""
    return np.where(np.asarray(tenor_months) <= 6, 0.3, 0.2)


def monthly_installment(outstanding, rate_pct, tenor_months):
    """Rumus cicilan dataset (bunga flat harian x 30 hari x tenor), cocok dengan kolom asli (selisih < Rp1)."""
    return outstanding * (1 + rate_pct / 100 * 30 * tenor_months) / tenor_months


def max_principal_ojk(income, tenor_months=24, rate_pct=None):
    """Pokok maksimum agar DSR = 30% pada tenor tertentu (bunga = batas OJK jika tidak diberikan)."""
    rate = ojk_rate_cap(tenor_months) if rate_pct is None else rate_pct
    return DSR_CAP * income * tenor_months / (1 + rate / 100 * 30 * tenor_months)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """5 rasio turunan + 4 flag kepatuhan OJK dari kolom mentah. Murni rumus per baris (tanpa statistik data)."""
    out = df.copy()
    out['debt_to_income_ratio'] = out['total_loan_outstanding_idr'] / out['monthly_income_idr']
    out['income_to_umr_ratio'] = out['monthly_income_idr'] / out['regional_minimum_wage_umr_idr']
    out['credit_limit_utilization_ratio'] = out['total_loan_outstanding_idr'] / out['credit_limit_granted_idr']
    out['monthly_installment_estimate_idr'] = monthly_installment(
        out['total_loan_outstanding_idr'], out['daily_interest_rate_pct'], out['loan_tenor_months'])
    out['debt_service_ratio'] = out['monthly_installment_estimate_idr'] / out['monthly_income_idr']
    out['flag_rate_above_ojk_cap'] = (out['daily_interest_rate_pct'] > ojk_rate_cap(out['loan_tenor_months'])).astype(int)
    out['flag_income_below_3jt'] = (out['monthly_income_idr'] < INCOME_MIN).astype(int)
    out['flag_apps_above_3'] = (out['num_active_lending_apps'] > APPS_MAX).astype(int)
    out['flag_dsr_above_30'] = (out['debt_service_ratio'] > DSR_CAP).astype(int)
    return out


DERIVED_NUM = ['debt_to_income_ratio', 'income_to_umr_ratio', 'credit_limit_utilization_ratio',
               'monthly_installment_estimate_idr', 'debt_service_ratio']


from sklearn.base import BaseEstimator, RegressorMixin, clone  # noqa: E402


class ClippedRegressor(RegressorMixin, BaseEstimator):
    """Pembungkus regressor: prediksi di-clip ke rentang DPD valid [0, 180].

    Ditaruh DI DALAM Pipeline supaya skor CV, tuning, test, aplikasi, dan Airflow
    semuanya memakai prediksi yang sama (model linear bisa memprediksi DPD negatif).
    Hyperparameter model di dalamnya diakses dengan prefix `reg__regressor__`.
    """

    def __init__(self, regressor=None, lower=0.0, upper=float(DPD_MAX)):
        self.regressor = regressor
        self.lower = lower
        self.upper = upper

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor).fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.regressor_.predict(X), self.lower, self.upper)


def predict_dpd(model, X: pd.DataFrame) -> np.ndarray:
    """Prediksi DPD dan clip ke rentang valid [0, 180]."""
    return np.clip(model.predict(X[RAW_INPUT]), 0, DPD_MAX)


def recommend(kelas: str, flag_dsr: int) -> str:
    text = ACTION_MAP[kelas]
    if int(flag_dsr) == 1:
        text += ' | Catatan: DSR melebihi batas OJK 30%'
    return text


# ---------------------------------------------------------------------------
# Simulasi restrukturisasi Before vs After (Model 2)
# ---------------------------------------------------------------------------
def simulate_restructuring(row: pd.DataFrame, model2, thr=DEFAULT_THRESHOLDS, tenor_options=(3, 6, 9, 12, 18, 24)):
    """Simulasi untuk 1 debitur (DataFrame 1 baris, kolom mentah).

    1. Untuk tiap tenor LEBIH PANJANG dari tenor sekarang:
       bunga = min(bunga awal, batas OJK tenor baru); cicilan & DSR dihitung ulang;
       Model 2 memprediksi DPD after.
    2. Opsi disarankan = opsi pertama dengan DSR <= 30% DAN prediksi DPD after <= 30 hari.
    3. Jika tidak ada: 'perlu penanganan lain' + pokok maksimum OJK (DSR = 30% pada tenor 24)
       dan skenario 'tenor 24 + pokok disesuaikan' yang dihitung ulang dengan Model 2.
    Return dict: options (DataFrame), best (Series/None), need_other (bool), haircut (dict/None)
    """
    base = row[RAW_INPUT].iloc[[0]].copy()
    cur_tenor = int(base['loan_tenor_months'].iloc[0])
    cur_rate = float(base['daily_interest_rate_pct'].iloc[0])
    rows = []
    for t in [t for t in tenor_options if t > cur_tenor]:
        sim = base.copy()
        sim['loan_tenor_months'] = t
        sim['daily_interest_rate_pct'] = min(cur_rate, float(ojk_rate_cap(t)))
        f = add_features(sim).iloc[0]
        dpd_after = float(predict_dpd(model2, sim)[0])
        rows.append({'tenor_bulan': t, 'bunga_harian_pct': float(sim['daily_interest_rate_pct'].iloc[0]),
                     'cicilan_bulanan_idr': round(float(f['monthly_installment_estimate_idr'])),
                     'dsr': round(float(f['debt_service_ratio']), 4),
                     'dsr_sesuai_ojk': bool(f['debt_service_ratio'] <= DSR_CAP),
                     'pred_dpd_after': round(dpd_after, 1),
                     'kolektibilitas_after': dpd_to_collectibility([dpd_after], thr)[0],
                     'memenuhi_syarat': bool(f['debt_service_ratio'] <= DSR_CAP and dpd_after <= thr['dpk'])})
    options = pd.DataFrame(rows)
    best = None
    if len(options) and options['memenuhi_syarat'].any():
        best = options[options['memenuhi_syarat']].iloc[0]
    haircut = None
    if best is None:
        income = float(base['monthly_income_idr'].iloc[0])
        pokok_maks = float(max_principal_ojk(income, 24))
        outstanding = float(base['total_loan_outstanding_idr'].iloc[0])
        sim = base.copy()
        sim['loan_tenor_months'] = 24
        sim['daily_interest_rate_pct'] = min(cur_rate, float(ojk_rate_cap(24)))
        sim['total_loan_outstanding_idr'] = min(outstanding, pokok_maks)
        f = add_features(sim).iloc[0]
        dpd_h = float(predict_dpd(model2, sim)[0])
        haircut = {'pokok_maks_ojk_idr': round(pokok_maks), 'pokok_sekarang_idr': round(outstanding),
                   'kelebihan_pokok_idr': round(max(0.0, outstanding - pokok_maks)),
                   'potongan_pct': round(max(0.0, 1 - pokok_maks / outstanding) * 100, 1),
                   'tenor_bulan': 24, 'cicilan_bulanan_idr': round(float(f['monthly_installment_estimate_idr'])),
                   'dsr': round(float(f['debt_service_ratio']), 4), 'pred_dpd_after': round(dpd_h, 1),
                   'kolektibilitas_after': dpd_to_collectibility([dpd_h], thr)[0]}
    return {'options': options, 'best': best, 'need_other': best is None, 'haircut': haircut}


def before_after_table(row: pd.DataFrame, model1, sim: dict, thr=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """Tabel Before vs After berdampingan (opsi disarankan, atau skenario pokok disesuaikan)."""
    base = row[RAW_INPUT].iloc[[0]]
    f = add_features(base).iloc[0]
    dpd_b = float(predict_dpd(model1, base)[0])
    before = {'Tenor (bulan)': int(f['loan_tenor_months']), 'Bunga harian (%)': float(f['daily_interest_rate_pct']),
              'Pokok (Rp)': round(float(f['total_loan_outstanding_idr'])),
              'Cicilan bulanan (Rp)': round(float(f['monthly_installment_estimate_idr'])),
              'DSR': f"{f['debt_service_ratio']:.1%}", 'Prediksi DPD (hari)': round(dpd_b, 1),
              'Kolektibilitas': dpd_to_collectibility([dpd_b], thr)[0], 'Status': dpd_to_eligibility([dpd_b], thr)[0]}
    if sim['best'] is not None:
        b = sim['best']
        after = {'Tenor (bulan)': int(b['tenor_bulan']), 'Bunga harian (%)': b['bunga_harian_pct'],
                 'Pokok (Rp)': before['Pokok (Rp)'], 'Cicilan bulanan (Rp)': int(b['cicilan_bulanan_idr']),
                 'DSR': f"{b['dsr']:.1%}", 'Prediksi DPD (hari)': b['pred_dpd_after'],
                 'Kolektibilitas': b['kolektibilitas_after'], 'Status': dpd_to_eligibility([b['pred_dpd_after']], thr)[0]}
        label = 'After (tenor disarankan)'
    else:
        h = sim['haircut']
        after = {'Tenor (bulan)': 24, 'Bunga harian (%)': min(before['Bunga harian (%)'], 0.2),
                 'Pokok (Rp)': min(before['Pokok (Rp)'], h['pokok_maks_ojk_idr']), 'Cicilan bulanan (Rp)': h['cicilan_bulanan_idr'],
                 'DSR': f"{h['dsr']:.1%}", 'Prediksi DPD (hari)': h['pred_dpd_after'],
                 'Kolektibilitas': h['kolektibilitas_after'], 'Status': dpd_to_eligibility([h['pred_dpd_after']], thr)[0]}
        label = 'After (tenor 24 + pokok disesuaikan)*'
    return pd.DataFrame({'Before': before, label: after})


SIM_DISCLAIMER = ('Simulasi menunjukkan perubahan prediksi model jika tenor atau pokok diubah. Model mempelajari '
                  'pola dari data, bukan hubungan sebab-akibat, sehingga hasil simulasi bukan jaminan debitur akan '
                  'membayar. Keputusan keringanan pokok tetap memerlukan persetujuan manajemen.')
SIM_NO_OPTION = 'Perpanjangan tenor tidak cukup — perlu penanganan lain (keringanan pokok/bunga atau eskalasi ke tim collection).'

# ---------------------------------------------------------------------------
# Label fitur untuk tampilan faktor SHAP
# ---------------------------------------------------------------------------
FEATURE_LABEL = {
    'age': 'Usia', 'monthly_income_idr': 'Penghasilan bulanan', 'regional_minimum_wage_umr_idr': 'UMR provinsi',
    'num_active_lending_apps': 'Jumlah aplikasi aktif', 'total_loan_outstanding_idr': 'Total pokok utang',
    'credit_limit_granted_idr': 'Limit kredit', 'loan_tenor_months': 'Tenor (bulan)',
    'daily_interest_rate_pct': 'Bunga harian (%)', 'debt_service_ratio': 'Rasio cicilan/penghasilan (DSR)',
    'flag_rate_above_ojk_cap': 'Bunga di atas batas OJK', 'flag_income_below_3jt': 'Penghasilan < Rp3 juta',
    'flag_apps_above_3': '> 3 aplikasi aktif', 'flag_dsr_above_30': 'DSR > 30%',
    'education_level': 'Pendidikan', 'employment_status': 'Pekerjaan', 'primary_loan_purpose': 'Tujuan pinjaman',
}


def base_feature(name: str) -> str:
    """'num__debt_service_ratio' -> 'debt_service_ratio'; 'cat__primary_loan_purpose_X' -> 'primary_loan_purpose'."""
    col = name.split('__', 1)[-1]
    for c in RAW_CAT:
        if col.startswith(c):
            return c
    return col


def fmt_value(col, v):
    if isinstance(v, str):
        return v
    if col.endswith('_idr'):
        return f'Rp{v:,.0f}'
    if col == 'debt_service_ratio':
        return f'{v:.0%}'
    if col == 'daily_interest_rate_pct':
        return f'{v:.2f}%'
    if col.startswith('flag_'):
        return 'ya' if v == 1 else 'tidak'
    return f'{v:g}'


def top_factors_from_shap(shap_row, feature_names, X_raw_row: pd.DataFrame, top=3):
    """Gabungkan nilai SHAP one-hot ke kolom asli, ambil `top` faktor yang paling MENAIKKAN prediksi DPD."""
    s = pd.Series(np.asarray(shap_row).ravel(), index=[base_feature(f) for f in feature_names]).groupby(level=0).sum()
    s = s.sort_values(ascending=False)
    feat = add_features(X_raw_row[RAW_INPUT]).iloc[0]
    return [{'fitur': k, 'label': FEATURE_LABEL.get(k, k), 'nilai': fmt_value(k, feat[k]), 'shap_hari': round(float(v), 1)}
            for k, v in s.head(top).items()]

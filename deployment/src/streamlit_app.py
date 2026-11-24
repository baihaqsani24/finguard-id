"""
========================================================
Final Project — FinGuard ID

Anggota Tim (Batch HCK-45):
- Alyssa Putri Azahra
- Baihaqsani
- Rizqullah Anjar Pradana

Entry point aplikasi Streamlit (hosting via streamlit.app).
========================================================
"""

import os
import streamlit as st
import eda
import prediction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(BASE_DIR, 'finguard_icon.png')
LOGO_PATH = os.path.join(BASE_DIR, 'finguard_logo.png')

st.set_page_config(
    page_title='FinGuard ID — Early-Warning Kolektibilitas Debitur',
    page_icon=ICON_PATH if os.path.exists(ICON_PATH) else '🛡️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Tampilkan ikon kecil di sidebar
if os.path.exists(ICON_PATH):
    st.sidebar.image(ICON_PATH, width=56)

st.sidebar.title('FinGuard ID')
st.sidebar.caption('Deteksi Dini DPD & Simulasi Restrukturisasi Hutang (Batch HCK-45)')

page = st.sidebar.selectbox(
    'Pilih Halaman',
    ['EDA', 'Prediction & Restrukturisasi', 'Batch CSV']
)

if page == 'EDA':
    eda.run()
elif page == 'Prediction & Restrukturisasi':
    prediction.run_single()
else:
    prediction.run_batch()

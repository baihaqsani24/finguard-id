# Panduan Setup Lokal Apache Airflow, PostgreSQL, Elasticsearch & Kibana — FinGuard ID

Panduan ini disusun khusus untuk menjalankan stack Docker lokal (`pipeline/airflow_ES.yaml`) di environment WSL2 / Docker Desktop kamu (mengacu pada setup Milestone 3 sebelumnya). Aplikasi **Streamlit** di-host terpisah di `streamlit.app`, sedangkan seluruh komponen di folder `pipeline/` dijalankan secara lokal.

---

## 1. Persiapan File `.env`
Masuk ke folder `pipeline/`, lalu salin `.env.example` menjadi `.env` (sesuaikan `AIRFLOW_UID` dengan hasil `id -u` di WSL jika perlu):

```bash
cd pipeline
cp .env.example .env
```

---

## 2. Jalankan Container Docker (`airflow_ES.yaml`)
Jalankan seluruh service (PostgreSQL port `5434`, Airflow Webserver port `8081`, Elasticsearch port `9200`, Kibana port `5601`):

```bash
docker compose -f airflow_ES.yaml up -d
```

Cek status container sampai `postgres`, `airflow-webserver`, `airflow-scheduler`, `elasticsearch`, dan `kibana` berstatus **Up (healthy)**:

```bash
docker compose -f airflow_ES.yaml ps
```

> **Catatan Penting:** Saat mematikan container, gunakan `docker compose -f airflow_ES.yaml stop` (bukan `down -v`) agar data PostgreSQL dan visualisasi Kibana tidak hilang.

---

## 3. Muat Data Mentah (`finguard_raw.csv`) ke PostgreSQL
1. Salin `data/finguard_raw.csv` dan `ddl.sql` ke container `postgres`:
   ```bash
   docker cp data/finguard_raw.csv postgres:/tmp/finguard_raw.csv
   docker cp ddl.sql postgres:/tmp/ddl.sql
   ```
2. Buat tabel `finguard_raw` dan muat 10.000 baris datanya:
   ```bash
   docker exec -i postgres psql -U airflow -d airflow -f /tmp/ddl.sql
   docker exec -i postgres psql -U airflow -d airflow -c "\copy finguard_raw FROM '/tmp/finguard_raw.csv' WITH (FORMAT csv, HEADER true);"
   docker exec -i postgres psql -U airflow -d airflow -c "SELECT COUNT(*) FROM finguard_raw;"
   ```
   Pastikan output `COUNT(*)` bernilai **`10000`**.

---

## 4. Penyesuaian Library di Container Airflow (Opsional untuk Scoring Level 2)
* **Level 1 (ETL + Validasi + Transformasi + Load ke Elasticsearch):** Langsung berjalan tanpa instalasi tambahan (`score_model` otomatis melakukan fallback aman ke Level 1 jika `scikit-learn` / `lightgbm` belum terpasang di image `apache/airflow:2.3.4`).
* **Level 2 (Aktifkan Batch Scoring `model_kelayakan.pkl` & `model_restrukturisasi.pkl` di Airflow):**
  1. Cek versi Python & library di container `airflow-scheduler`:
     ```bash
     docker exec -u airflow airflow-scheduler python -c "import sys; print(sys.version)"
     ```
  2. Pasang versi library yang sama dengan `model_metadata.json` (`scikit-learn==1.6.1`, `lightgbm==4.6.0`, `joblib==1.4.2`, `elasticsearch==7.13.4`) di container `airflow-scheduler` dan `airflow-webserver`:
     ```bash
     docker exec -u airflow airflow-scheduler pip install scikit-learn==1.6.1 lightgbm==4.6.0 joblib==1.4.2 elasticsearch==7.13.4
     docker exec -u airflow airflow-webserver pip install scikit-learn==1.6.1 lightgbm==4.6.0 joblib==1.4.2 elasticsearch==7.13.4
     ```
  *(Jika `libgomp.so.1` dibutuhkan oleh LightGBM di dalam container, jalankan `docker exec -u root airflow-scheduler bash -c "apt-get update && apt-get install -y libgomp1"`.)*

---

## 5. Trigger DAG & Cek Hasil di Kibana
1. Buka Airflow Webserver di browser: **`http://localhost:8081`** (login default: `airflow` / `airflow`).
2. Aktifkan DAG **`FinGuard_ID_DAG`** lalu klik **Trigger DAG** (`fetch_psql >> validate_data >> transform_features >> score_model >> insert_to_elastic`).
3. Verifikasi jumlah dokumen di Elasticsearch tetap 10.000 (idempotent):
   ```bash
   curl http://localhost:9200/finguard_debitur/_count
   ```
4. Buka Kibana di **`http://localhost:5601`**, buat *Index Pattern* **`finguard_debitur`** dengan *time field* `snapshot_date` (pastikan *time range* mencakup hari ini).
5. Bangun visualisasi dashboard (gunakan **Average** untuk rata-rata `days_past_due` / `pred_dpd`, dan filter **`split: test`** khusus untuk panel evaluasi akurasi model).
6. Ambil *screenshot* panel dashboard Kibana bila ingin mendokumentasikan hasilnya (misalnya ke folder **`images/`** di root repositori).

---

## 6. Validasi Great Expectations (`gx_validation.ipynb`)
Jalankan `pipeline/gx_validation.ipynb` pada environment lokal `h8_env` (Great Expectations `0.18.19`) untuk memverifikasi 10 *expectations* pada `finguard_raw.csv`.

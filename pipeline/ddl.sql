-- =====================================================================
-- FinGuard ID — DDL raw layer (PostgreSQL)
-- Tabel menyimpan 15 kolom MENTAH data debitur (target = days_past_due;
-- 4 kolom generatif LLM TIDAK disimpan) seperti di database
-- operasional. Kolom turunan (DTI, DSR, flag OJK) dihitung oleh DAG
-- Airflow lewat features.py, bukan disimpan di sini.
--
-- Keputusan:
--   * Nama kolom dikutip ganda -> nama persis sama dengan header CSV.
--   * Tanpa PRIMARY KEY (raw layer): data mentah dimuat apa adanya;
--     keunikan borrower_id dicek di task `validate` dan di Great Expectations.
--   * Rupiah = BIGINT (nilai sampai puluhan juta, aman dan tanpa pembulatan).
-- =====================================================================

-- 1) Buat database (jalankan sekali dari psql sebagai user airflow)
-- CREATE DATABASE finguard;

-- 2) Buat tabel (di database finguard)
DROP TABLE IF EXISTS finguard_raw;

CREATE TABLE finguard_raw (
    "borrower_id"                      VARCHAR(30),
    "age"                              INTEGER,
    "monthly_income_idr"               BIGINT,
    "regional_minimum_wage_umr_idr"    BIGINT,
    "num_active_lending_apps"          INTEGER,
    "total_loan_outstanding_idr"       BIGINT,
    "credit_limit_granted_idr"         BIGINT,
    "loan_tenor_months"                INTEGER,
    "daily_interest_rate_pct"          NUMERIC(5, 3),
    "gender"                           VARCHAR(10),
    "province"                         VARCHAR(50),
    "education_level"                  VARCHAR(30),
    "employment_status"                VARCHAR(40),
    "primary_loan_purpose"             VARCHAR(50),
    "days_past_due"                    INTEGER
);

-- Urutan kolom HARUS sama dengan header pipeline/data/finguard_raw.csv (COPY ... CSV HEADER memetakan per posisi)

-- 3) Isi data (sama dengan Milestone 3):
--    docker cp pipeline/data/finguard_raw.csv postgres:/tmp/finguard_raw.csv
--    docker exec -it postgres psql -U airflow -d finguard
COPY finguard_raw FROM '/tmp/finguard_raw.csv' DELIMITER ',' CSV HEADER;

-- 4) Cek
SELECT COUNT(*) AS jumlah_baris FROM finguard_raw;                -- harus 10000
SELECT MIN("days_past_due"), MAX("days_past_due"), AVG("days_past_due") FROM finguard_raw;   -- 0 | 180 | ±51,8

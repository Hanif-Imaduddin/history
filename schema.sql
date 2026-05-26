-- ClarioAI PostgreSQL Schema
-- Jalankan file ini untuk membuat database: psql -U postgres -d clario_ai -f schema.sql

-- Tabel users
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role          VARCHAR(20) NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tabel device_fingerprints: satu fingerprint hanya bisa milik satu user
CREATE TABLE IF NOT EXISTS device_fingerprints (
    id               SERIAL PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    fingerprint_hash VARCHAR(255) NOT NULL,
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(fingerprint_hash)
);

-- Tabel analysis_logs: rekam setiap kali user memulai analisis
CREATE TABLE IF NOT EXISTS analysis_logs (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER REFERENCES users(id) ON DELETE SET NULL,
    username   VARCHAR(50),
    session_id VARCHAR(100),
    sector     VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tabel auth_logs: rekam event registrasi & login (termasuk flag fingerprint duplikat)
CREATE TABLE IF NOT EXISTS auth_logs (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER REFERENCES users(id) ON DELETE SET NULL,
    username   VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- 'register', 'register_fp_duplicate', 'login', 'login_failed'
    details    TEXT,
    ip_address VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indeks untuk performa query
CREATE INDEX IF NOT EXISTS idx_analysis_logs_user_date ON analysis_logs(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_auth_logs_event        ON auth_logs(event_type, created_at);
CREATE INDEX IF NOT EXISTS idx_device_fp_user         ON device_fingerprints(user_id);

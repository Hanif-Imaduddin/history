# Clario.AI — Sistem Pendukung Keputusan Berbasis Agentic AI untuk Perencanaan Bisnis Kewirausahaan

## Latar Belakang dan Tujuan

Kegagalan usaha baru merupakan fenomena yang marak terjadi di berbagai belahan dunia. Data dari U.S. Bureau of Labor Statistics mencatat bahwa 20,4% bisnis baru gagal pada tahun pertama dan 49,4% gagal di tahun kelima. Faktor utama penyebab kegagalan tersebut mencakup kehabisan modal, ketidaksesuaian produk dengan pasar, serta perencanaan bisnis yang tidak dilandasi analisis data yang memadai. Di sisi lain, Indonesia mencatat pertumbuhan wirausaha yang signifikan dengan 53,38 juta wirausaha aktif pada tahun 2025, namun pertumbuhan kuantitatif tersebut belum sepenuhnya mencerminkan tingkat keberlanjutan dan kualitas perencanaan yang memadai.

ClarioAI hadir sebagai solusi berupa _Decision Support System_ berbasis _Agentic AI_ yang dirancang khusus untuk mendukung proses perencanaan bisnis kewirausahaan secara menyeluruh. Sistem ini mengintegrasikan _Large Language Model_ (LLM) dengan arsitektur multi-agent yang mampu melakukan penalaran mandiri, riset pasar secara _real-time_, pemodelan finansial, analisis strategis, hingga pengawasan etika dan kepatuhan hukum. Dengan pendekatan ini, wirausahawan dapat memperoleh rencana bisnis yang komprehensif dan berbasis data dalam waktu yang jauh lebih singkat dibandingkan pendekatan konvensional.

Sistem ini dibangun di atas paradigma _Tree of Thoughts_ (ToT) yang memungkinkan eksplorasi berbagai jalur perencanaan bisnis sebelum memilih solusi yang paling optimal. Setiap agen beroperasi secara otonom namun terkoordinasi melalui orkestrasi terpusat, menghasilkan laporan terstruktur yang mencakup analisis pasar, Lean Canvas, proyeksi keuangan berbasis simulasi Monte-Carlo, serta validasi etika dan regulasi. Sistem juga mendukung mekanisme _human-in-the-loop_ di mana pengguna dapat memberikan umpan balik untuk memandu iterasi analisis berikutnya.

ClarioAI dikembangkan oleh Kelompok 11 sebagai bagian dari proyek mata kuliah Capstone Project dengan rentang pengembangan Maret hingga Mei 2026. Sistem ini berpotensi menjadi katalisator penting dalam ekosistem kewirausahaan Indonesia, menjembatani kesenjangan antara ketersediaan data pasar yang melimpah dengan kapasitas analitis wirausahawan yang terbatas.

---

## Arsitektur Sistem

### Pola Multi-Agent dengan Supervisor

ClarioAI menggunakan pola _supervisor multi-agent_ yang diorkestrasikan menggunakan kerangka kerja **LangGraph**. Seluruh alur kerja dikontrol oleh satu agen pusat (_Lead Orchestrator_) yang mendistribusikan tugas ke empat agen spesialis secara berurutan, kemudian mengevaluasi hasil dari seluruh agen untuk menentukan apakah rencana bisnis layak diterima atau perlu direvisi.

### Alur Graph

![Alur Graph ClarioAI](Graph%20Flow.svg)

Orkestrasi bersifat iteratif; jika `approval_status` masih `rejected` dan jumlah iterasi belum mencapai `max_iterations`, pipeline akan kembali dijalankan dari Market Scout hingga seluruh laporan direvisi dan dievaluasi ulang.

### Lima Agen Utama

**1. Lead Orchestrator**

Agen pusat yang bertanggung jawab atas dekomposisi tujuan bisnis, manajemen siklus iterasi, dan evaluasi akhir. Agen ini membaca seluruh laporan dari keempat agen spesialis, kemudian menetapkan `approval_status` menjadi `approved`, `rejected`, atau `pending`, disertai `orchestrator_feedback` yang memuat catatan perbaikan spesifik untuk iterasi berikutnya.

**2. Market Scout Agent**

Melakukan riset pasar secara _real-time_ menggunakan alat pencarian internet yang terintegrasi dengan BrightData SERP API. Agen ini mengidentifikasi peluang pasar, tren industri, perilaku konsumen, dan lanskap kompetitor, kemudian menghasilkan `MarketScoutReport` berisi daftar ide bisnis beserta penjelasan kontekstualnya.

**3. Strategic Architect Agent**

Menyusun kerangka strategis bisnis berdasarkan laporan pasar yang telah dihasilkan. Keluaran agen ini adalah `StrategicReport` yang mencakup analisis SWOT (_Strengths, Weaknesses, Opportunities, Threats_) dan analisis PESTEL (_Political, Economic, Social, Technological, Environmental, Legal_) secara mendalam.

**4. Financial Analyst Agent**

Melakukan pemodelan keuangan yang komprehensif berdasarkan konteks pasar dan strategi bisnis. Agen ini menghasilkan `FinancialAnalysisReport` yang mencakup estimasi biaya awal, proyeksi pendapatan, skenario risiko, analisis titik impas (_break-even_), serta simulasi alur kas untuk membantu pengambilan keputusan berbasis data finansial.

**5. Ethics Guardian Agent**

Mengevaluasi seluruh rencana bisnis dari perspektif etika dan kepatuhan hukum, khususnya regulasi yang berlaku di Indonesia. Agen ini menghasilkan `EthicsAnalysisReport` yang mengidentifikasi potensi risiko hukum, persyaratan perizinan, dan rekomendasi mitigasi untuk memastikan rencana bisnis beroperasi dalam koridor yang legal dan etis.

### Manajemen State

Seluruh data sesi dikelola melalui `EBPState`, sebuah `TypedDict` LangGraph yang mengakumulasi keluaran setiap agen dalam satu struktur state terpusat.

| Field | Tipe | Keterangan |
|-------|------|------------|
| `state_id` | `str` | Identifikasi unik sesi (UUID) |
| `user_id` | `str` | Identifikasi pengguna |
| `bussiness_constraints` | `BussinessConstraints` | Input awal dari pengguna |
| `market_scout_report` | `MarketScoutReport` | Laporan riset pasar |
| `strategic_report` | `StrategicReport` | Laporan analisis strategis |
| `financial_analysis_report` | `FinancialAnalysisReport` | Laporan analisis keuangan |
| `ethics_analysis_report` | `EthicsAnalysisReport` | Laporan etika dan kepatuhan |
| `approval_status` | `pending / approved / rejected` | Status persetujuan saat ini |
| `orchestrator_feedback` | `str` | Catatan perbaikan dari orchestrator |
| `messages` | `List[BaseMessage]` | Riwayat pesan LangGraph |
| `iteration` | `int` | Nomor iterasi saat ini |
| `max_iterations` | `int` | Batas maksimum iterasi |
| `user_feedback` | `str` | Umpan balik dari pengguna |

State disimpan secara persisten di MongoDB sehingga sesi dapat dilanjutkan kapan saja menggunakan `state_id`.

### Alat (Tools)

**internet_search** — Alat pencarian internet yang digunakan oleh Market Scout Agent. Alat ini mengirimkan kueri ke Google Search melalui BrightData SERP API, mengambil lima URL teratas, lalu melakukan _web scraping_ dan segmentasi teks (_chunking_) secara otomatis. Relevansi setiap segmen teks dinilai menggunakan model _embedding_ Qwen melalui perbandingan kesamaan kosinus (_cosine similarity_), sehingga hanya konten paling relevan yang diteruskan ke agen.

### Struktur Direktori

```
ClarioAI/
├── states/
│   └── schema.py           # EBPState dan seluruh dataclass laporan
├── tools/
│   └── internet_search.py  # Alat pencarian internet (BrightData + embedding)
├── nodes/
│   ├── lead_orchestrator.py
│   ├── market_scout.py
│   ├── strategic_architect.py
│   ├── financial_analyst.py
│   └── ethics_agent.py
├── graphs/
│   └── ebp_graph.py        # Definisi dan kompilasi graph LangGraph
├── functions/
│   ├── llm.py              # Inisialisasi LLM (DeepInfra / Qwen)
│   ├── mongodb.py          # Fungsi CRUD state ke MongoDB
│   └── agent_utils.py      # Utilitas bersama (extract_json, format_constraints)
├── models/
├── test_system.ipynb       # Notebook pengujian komponen
├── searching_test.ipynb    # Notebook pengujian pencarian
└── requirements.txt
```

---

## Requirements dan Teknologi

### Bahasa Pemrograman

- **Python 3.12.9**

### Dependensi Python

Instal seluruh dependensi menggunakan:

```bash
pip install -r requirements.txt
```

| Paket | Versi Minimum | Kegunaan |
|-------|--------------|----------|
| `langchain` | 0.3.0 | Abstraksi agen dan alat LangChain |
| `langchain-openai` | 0.2.0 | Integrasi LLM berbasis OpenAI-compatible API |
| `langchain-core` | 0.3.0 | Komponen inti LangChain (pesan, alat) |
| `langgraph` | 0.2.0 | Orkestrasi graph multi-agent |
| `pymongo` | 4.7.0 | Driver MongoDB untuk persistensi state |
| `requests` | 2.31.0 | Klien HTTP untuk panggilan API eksternal |
| `urllib3` | 2.0.0 | Dependensi HTTP tingkat rendah |
| `typing-extensions` | 4.9.0 | Back-port tipe data Python |
| `numpy` | - | Komputasi vektor untuk _similarity search_ |
| `beautifulsoup4` | - | Parsing HTML saat _web scraping_ |

### API Eksternal

**DeepInfra API**

Digunakan untuk inferensi LLM dan komputasi _embedding_ teks.

- Model LLM: `Qwen/Qwen3.5-397B-A17B`
- Model Embedding: `Qwen/Qwen3-Embedding-8B`
- Base URL: `https://api.deepinfra.com/v1/openai`

**BrightData SERP API**

Digunakan oleh alat `internet_search` untuk mengambil hasil pencarian Google secara terprogram.

- Zona: `serp_api_capstone`
- Endpoint: `https://api.brightdata.com/request`

**MongoDB**

Digunakan untuk persistensi state sesi secara lokal.

- URI default: `mongodb://localhost:27017`
- Nama database: `clario_ai`

Pastikan MongoDB berjalan di mesin lokal sebelum menjalankan sistem.

### Cara Menjalankan
Coming soon!!

---

## Notebook Pengujian

File `test_system.ipynb` adalah notebook Jupyter yang digunakan untuk menguji seluruh komponen sistem secara terstruktur dan terpisah sebelum diintegrasikan dalam alur produksi. Notebook ini terdiri dari sembilan seksi pengujian:

| Seksi | Nama | Deskripsi |
|-------|------|-----------|
| 1 | Imports & Environment | Memverifikasi ketersediaan semua paket yang dibutuhkan |
| 2 | State Schema | Menguji pembuatan dataclass `BussinessConstraints`, seluruh dataclass laporan, dan `EBPState` |
| 3 | Utility Functions | Menguji fungsi `extract_json` (parsing JSON dari respons LLM) dan `format_constraints` |
| 4 | LLM Connectivity | Menguji koneksi ke DeepInfra API dan memverifikasi model Qwen dapat merespons |
| 5 | Internet Search Tool | Menguji alat `internet_search` terhadap BrightData SERP API dengan kueri nyata |
| 6 | MongoDB | Menguji koneksi, operasi `create_new_state`, `save_state`, dan `load_state`, termasuk serialisasi pesan |
| 7 | Agent Nodes | Menguji setiap agen secara mandiri dengan state minimal: Market Scout, Strategic Architect, Financial Analyst, Ethics Analyst, dan Lead Orchestrator |
| 8 | Graph Pipeline | Menguji kompilasi graph LangGraph dan logika routing kondisional (`_route_from_orchestrator`) secara unit |
| 9 | End-to-End | Menjalankan pipeline lengkap seluruh lima agen dengan skenario bisnis nyata (`max_iterations=1`) untuk memvalidasi integrasi sistem secara menyeluruh |

Pengujian end-to-end pada seksi 9 menggunakan skenario bisnis platform micro-learning berbasis video pendek untuk persiapan SNBT dan sertifikasi profesional di Indonesia, yang merupakan skenario representatif untuk memvalidasi kemampuan sistem dalam menghasilkan rencana bisnis yang komprehensif dan koheren dari semua agen secara bersamaan.

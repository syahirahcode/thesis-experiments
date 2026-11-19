# Backend Makeup Analysis

Backend for the thesis makeup-analysis experiments.

[English](#english) | [Bahasa Indonesia](#bahasa-indonesia)

---

## English

### Setup from a fresh checkout (Windows / PowerShell)

Run the commands below from this `backend/` directory.

#### 1. Check Python

The backend environment used for this project runs Python 3.13.0. Use Python
3.13 to reproduce that setup:

```powershell
python --version
```

#### 2. Create and activate a virtual environment

Create the environment once. Activate it whenever you open a new terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow scripts for the current terminal only
with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate
again. Alternatively, run commands with `.\.venv\Scripts\python.exe` without
activating the environment.

#### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include PyTorch/torchvision, OpenCV, MediaPipe, NumPy, and
the application and reporting libraries. To run the test suite, install pytest
in this environment (`python -m pip install pytest`). Pytest is not required
for running the evaluation.

#### 4. Create and configure `.env`

Copy the example file only if `.env` does not already exist. If `.env` exists,
keep it and edit it instead of overwriting local settings:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

The checked-in example selects **Vikey.ai** with `LLM_OPTION=vikey`. Set
`VIKEY_KEY` to your own Vikey API key before running the evaluation. Never
commit `.env` or put a real key in source files.

Relevant `.env` settings for RQ2/RQ3:

| Variable | Required / default | Purpose |
|---|---|---|
| `LLM_OPTION` | `vikey` in `.env.example` | Selects `vikey` or `gemini`. Set this explicitly; if `.env` is absent, the application defaults to `gemini`. |
| `VIKEY_KEY` | Required for Vikey | Your Vikey API key. |
| `VIKEY_URL` | `https://api.vikey.ai/v1` | Vikey OpenAI-compatible endpoint. |
| `VIKEY_MODEL` | `anthropic/haiku-5.5` in `.env.example` | Model ID enabled for your Vikey account. |
| `LLM_TIMEOUT_SECONDS` | `180` in `.env.example` | Request timeout. |
| `LLM_MAX_RETRIES` | `1` in `.env.example` | Retry limit for transient LLM errors. |
| `LLM_MAX_ROUNDS` | `8` in application defaults | Maximum structured tool-call rounds. |
| `LLM_TEMPERATURE` | `0` in application defaults | LLM sampling temperature. |
| `ATTRIBUTE_SOURCE` | `cnn` in application defaults | `cnn` runs Model B; `ground_truth` uses expert labels for an ablation. |
| `PRODUCT_SOURCE` | `json` in `.env.example` | `json` reads the local product catalog. `postgres` requires a Supabase connection. |
| `TOP_K_PRODUCTS_PER_CATEGORY` | `4` in `.env.example` | Number of products searched per category. |
| `CNN_DEVICE` | `auto` in application defaults | CNN inference device: `auto`, `cpu`, or `cuda`. |
| `CNN_MODEL_PATH` | `app/models/best_model_b_rgb.pth` | Model B checkpoint path. |
| `KB_RULES_PATH` | `app/core/kb_rules.json` | Season knowledge base path. |
| `PRODUCTS_PATH` | `app/core/products.json` | Local product catalog path when using JSON. |
| `ASSETS_DIR` | `assets` | Root directory for report images. |
| `EXPECTED_N_CASES` | `24` in application defaults | Expected number of test images. Change only when the experiment dataset is intentionally different. |

To use **Gemini instead of Vikey**, set `LLM_OPTION=gemini` and provide
`GEMINI_API_KEY`. `LLM_MODEL` selects the Gemini model; `VIKEY_KEY` and
`VIKEY_MODEL` are not used in that mode.

If creating `.env` manually instead of copying `.env.example`, the minimum
RQ2/RQ3 Vikey configuration is:

```dotenv
LLM_OPTION=vikey
VIKEY_URL=https://api.vikey.ai/v1
VIKEY_KEY=your_vikey_api_key
VIKEY_MODEL=anthropic/haiku-5.5
PRODUCT_SOURCE=json
ATTRIBUTE_SOURCE=cnn
```

Keep the other path and timeout settings from `.env.example` unless you need to
override them. `.env` is loaded automatically from the `backend/` directory.

The example also contains `OPENROUTER_*`, `SUPABASE_URL`, `SUPABASE_KEY`,
`SUPABASE_BUCKET`, and `VITE_API_URL`. They are not required for the default
RQ2/RQ3 run (`PRODUCT_SOURCE=json`, LLM via Vikey). For
`PRODUCT_SOURCE=postgres`, set `SUPABASE_CONNECTION_STRING` to the database
connection string; the catalog is then read from `public.products`.
Supabase URL/key/bucket settings are for storage and product-upload workflows,
not the default evaluation.

#### 5. Check the experiment inputs

The runner expects the following paths relative to `backend/`:

- `app/core/kb_rules.json`: season knowledge base.
- `app/core/products.json`: product catalog when `PRODUCT_SOURCE=json`.
- `app/models/best_model_b_rgb.pth`: Model B checkpoint when
  `ATTRIBUTE_SOURCE=cnn`.
- `experiments/test_data/`: the 24 evaluation images by default.
- `experiments/Final Dataset - Data.csv`: ground-truth labels; image filenames
  must match files in `test_data/`.
- `assets/looks/`, `assets/saturations/`, and `assets/products/`: report and
  product image assets.

Model checkpoint files are not tracked by Git. Download the Model B checkpoint
from the [shared Google Drive folder](https://drive.google.com/drive/folders/1LU02WuQNT-dqOiW8pKUcBu4ID9lIZrHC?usp=drive_link)
and place `best_model_b_rgb.pth` in `app/models/`. Ensure the final path is
`app/models/best_model_b_rgb.pth`.

The expected image count defaults to 24 (`EXPECTED_N_CASES`). The runner checks
the paths, image count, and experiment design before calling an LLM. MediaPipe
uses `face_landmarker.task` from the backend working directory and downloads it
automatically if it is missing, so the first run needs network access.

#### 6. Preview, then run RQ2/RQ3

Validate the setup and preview a small run without calling agents:

```powershell
python experiments/run_eval_rq2_rq3.py --limit 2 --modes multi_agent single_agent --dry-run
```

### Verify Model B preprocessing and labels

The checker loads every image referenced by the ground-truth CSV and reports
per-attribute agreement:

```powershell
python experiments/check_rq1_model_b.py
```

It uses the RQ1 Model B label mappings and inference preprocessing
(BGR-to-RGB, 224×224 resize, float32 scaling to `[0, 1]`, then CHW tensor
layout).

### Run the paired RQ2/RQ3 evaluation

Run all 24 cases with both architectures (48 runs):

```powershell
python experiments/run_eval_rq2_rq3.py
```

To run a shorter subset, pass `--limit N` (or the compatible `--cases N` alias).
The selected cases are the first N images in sorted filename order; omitting
the option runs the complete dataset. To select cases by their numeric IDs
from the ground-truth CSV, pass them to `--case-id`; this option is mutually
exclusive with `--limit`. Select one or both architectures with
`--modes multi_agent single_agent`, preview the planned run count with
`--dry-run`, and continue an interrupted evaluation with `--resume`:

```powershell
python experiments/run_eval_rq2_rq3.py --limit 2 --modes multi_agent single_agent --dry-run
python experiments/run_eval_rq2_rq3.py --case-id 510 515 --modes multi_agent single_agent --dry-run
python experiments/run_eval_rq2_rq3.py --limit 2 --resume
```

The runner creates its output directory automatically. Checkpoints, traces,
summary metrics, and HTML reports are saved there; this directory is excluded
from Git. Copy results you need to preserve before deleting it. Use `--resume`
only if a checkpoint from the interrupted run is still present.

The default `ATTRIBUTE_SOURCE=cnn` runs Model B independently for each
architecture and each case. Multi-Agent runs its own
`FacialAgent → ReasoningAgent → ProductAgent` pipeline through the orchestrator.
Single-Agent independently preprocesses the image using MediaPipe facial ROI
extraction, runs its own CNN instance, and then makes one structured LLM request
for season reasoning, the complete explanation, and product picks. The LLM uses
the shared catalog-search tool during its structured function-call loop. It
does not construct or call any other agent and never consumes Multi-Agent
outputs. `ATTRIBUTE_SOURCE=ground_truth` can be used as an expert-label
ablation. Both modes use the same checkpoint, knowledge base, product catalog,
and evaluation metrics, but their image-inference executions and outputs are
isolated.

Both LLM architectures choose the primary and sister season from the supplied
attributes and knowledge base; neither uses a calculated season score or
ranking. The HTML report displays the primary season more prominently, the
optional sister season below it, and each makeup palette's HEX code.

The runner validates the full data design, prints the total planned runs
(active cases multiplied by selected modes), and writes each case/mode result
to an output checkpoint keyed by case ID and mode. A resumed run skips
completed keys. Trace logs and summary CSVs contain case-level aggregates. The
successful Multi-Agent result for each case generates one HTML report
immediately; Single-Agent does not generate a second report. Each newly
completed case/mode is also appended to a run-specific progress trace in the
output directory. Reports use backend-configured attribute and season swatches
in `app/core/config.py`; makeup palette HEX codes are printed beside their
swatches.

Both product stages use the same `search_products` tool implementation. The LLM
requests a season, category, and limit; visual attributes are bound server-side,
and the tool does not replace an incorrect season argument. ProductAgent uses
a separate LLM stage in Multi-Agent, while Single-Agent calls the same tool
from its single reasoning/product/explanation loop. The number of search calls
made with a season different from the final reasoning season is reported per
mode as `wrong_season_search_calls`.

---

## Bahasa Indonesia

### Persiapan dari checkout baru (Windows / PowerShell)

Jalankan perintah-perintah berikut dari folder `backend/`.

#### 1. Periksa Python

Environment backend yang digunakan untuk proyek ini berjalan dengan Python
3.13.0. Gunakan Python 3.13 untuk mereproduksi setup tersebut:

```powershell
python --version
```

#### 2. Buat dan aktifkan virtual environment

Buat environment satu kali. Aktifkan kembali setiap kali membuka terminal baru:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Jika PowerShell memblokir aktivasi, izinkan script hanya untuk terminal saat
ini dengan `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, lalu
aktifkan kembali. Alternatifnya, jalankan perintah menggunakan
`.\.venv\Scripts\python.exe` tanpa mengaktifkan environment.

#### 3. Instal dependensi

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Daftar dependensi mencakup PyTorch/torchvision, OpenCV, MediaPipe, NumPy,
serta library aplikasi dan pembuatan laporan. Untuk menjalankan test suite,
instal pytest di environment ini (`python -m pip install pytest`). Pytest tidak
diperlukan untuk menjalankan evaluasi.

#### 4. Buat dan konfigurasi `.env`

Salin file contoh hanya jika `.env` belum ada. Jika `.env` sudah tersedia,
pertahankan dan edit file tersebut agar pengaturan lokal tidak tertimpa:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

File contoh yang ada memilih **Vikey.ai** melalui `LLM_OPTION=vikey`. Isi
`VIKEY_KEY` dengan API key Vikey milik Anda sebelum menjalankan evaluasi.
Jangan commit `.env` atau menaruh API key asli di file sumber.

Pengaturan `.env` yang relevan untuk RQ2/RQ3:

| Variabel | Wajib / default | Fungsi |
|---|---|---|
| `LLM_OPTION` | `vikey` di `.env.example` | Memilih `vikey` atau `gemini`. Atur secara eksplisit; jika `.env` tidak ada, default aplikasi adalah `gemini`. |
| `VIKEY_KEY` | Wajib untuk Vikey | API key Vikey Anda. |
| `VIKEY_URL` | `https://api.vikey.ai/v1` | Endpoint Vikey yang kompatibel dengan OpenAI API. |
| `VIKEY_MODEL` | `anthropic/haiku-5.5` di `.env.example` | ID model yang tersedia untuk akun Vikey Anda. |
| `LLM_TIMEOUT_SECONDS` | `180` di `.env.example` | Batas waktu request. |
| `LLM_MAX_RETRIES` | `1` di `.env.example` | Batas percobaan ulang untuk error LLM sementara. |
| `LLM_MAX_ROUNDS` | `8` pada default aplikasi | Batas putaran structured tool-call. |
| `LLM_TEMPERATURE` | `0` pada default aplikasi | Sampling temperature LLM. |
| `ATTRIBUTE_SOURCE` | `cnn` pada default aplikasi | `cnn` menjalankan Model B; `ground_truth` memakai label ahli untuk eksperimen ablation. |
| `PRODUCT_SOURCE` | `json` di `.env.example` | `json` membaca katalog produk lokal. `postgres` memerlukan koneksi Supabase. |
| `TOP_K_PRODUCTS_PER_CATEGORY` | `4` di `.env.example` | Jumlah produk yang dicari per kategori. |
| `CNN_DEVICE` | `auto` pada default aplikasi | Perangkat inferensi CNN: `auto`, `cpu`, atau `cuda`. |
| `CNN_MODEL_PATH` | `app/models/best_model_b_rgb.pth` | Path checkpoint Model B. |
| `KB_RULES_PATH` | `app/core/kb_rules.json` | Path knowledge base season. |
| `PRODUCTS_PATH` | `app/core/products.json` | Path katalog produk lokal saat memakai JSON. |
| `ASSETS_DIR` | `assets` | Folder utama gambar laporan. |
| `EXPECTED_N_CASES` | `24` pada default aplikasi | Jumlah gambar uji yang diharapkan. Ubah hanya jika dataset eksperimen memang berbeda. |

Untuk memakai **Gemini, bukan Vikey**, atur `LLM_OPTION=gemini` dan isi
`GEMINI_API_KEY`. `LLM_MODEL` memilih model Gemini; `VIKEY_KEY` dan
`VIKEY_MODEL` tidak digunakan pada mode tersebut.

Jika membuat `.env` secara manual, bukan menyalin `.env.example`, konfigurasi
minimum Vikey untuk RQ2/RQ3 adalah:

```dotenv
LLM_OPTION=vikey
VIKEY_URL=https://api.vikey.ai/v1
VIKEY_KEY=your_vikey_api_key
VIKEY_MODEL=anthropic/haiku-5.5
PRODUCT_SOURCE=json
ATTRIBUTE_SOURCE=cnn
```

Pertahankan pengaturan path dan timeout lainnya dari `.env.example`, kecuali
memang perlu mengubahnya. `.env` dimuat otomatis dari folder `backend/`.

File contoh juga memuat `OPENROUTER_*`, `SUPABASE_URL`, `SUPABASE_KEY`,
`SUPABASE_BUCKET`, dan `VITE_API_URL`. Variabel tersebut tidak diperlukan
untuk run RQ2/RQ3 default (`PRODUCT_SOURCE=json`, LLM melalui Vikey). Untuk
`PRODUCT_SOURCE=postgres`, isi `SUPABASE_CONNECTION_STRING` dengan connection
string database; katalog kemudian dibaca dari `public.products`.
Pengaturan Supabase URL/key/bucket digunakan untuk workflow storage dan upload
produk, bukan untuk evaluasi default.

#### 5. Periksa input eksperimen

Runner mengharapkan path berikut relatif terhadap folder `backend/`:

- `app/core/kb_rules.json`: knowledge base season.
- `app/core/products.json`: katalog produk saat `PRODUCT_SOURCE=json`.
- `app/models/best_model_b_rgb.pth`: checkpoint Model B saat
  `ATTRIBUTE_SOURCE=cnn`.
- `experiments/test_data/`: secara default berisi 24 gambar evaluasi.
- `experiments/Final Dataset - Data.csv`: label ground truth; nama file gambar
  harus cocok dengan file di `test_data/`.
- `assets/looks/`, `assets/saturations/`, dan `assets/products/`: aset gambar
  laporan dan produk.

File checkpoint model tidak dilacak Git. Unduh checkpoint Model B dari
[folder Google Drive bersama](https://drive.google.com/drive/folders/1LU02WuQNT-dqOiW8pKUcBu4ID9lIZrHC?usp=drive_link),
lalu letakkan `best_model_b_rgb.pth` di `app/models/`. Pastikan path akhirnya
adalah `app/models/best_model_b_rgb.pth`.

Jumlah gambar yang diharapkan default-nya 24 (`EXPECTED_N_CASES`). Sebelum
memanggil LLM, runner memeriksa path, jumlah gambar, dan desain eksperimen.
MediaPipe menggunakan `face_landmarker.task` dari working directory
`backend/` dan akan mengunduh file tersebut otomatis jika belum ada. Karena itu,
run pertama memerlukan akses jaringan.

#### 6. Preview, lalu jalankan RQ2/RQ3

Validasi setup dan preview run kecil tanpa memanggil agent:

```powershell
python experiments/run_eval_rq2_rq3.py --limit 2 --modes multi_agent single_agent --dry-run
```

### Verifikasi preprocessing dan label Model B

Checker memuat setiap gambar yang dirujuk oleh ground-truth CSV dan melaporkan
kesesuaian per atribut:

```powershell
python experiments/check_rq1_model_b.py
```

Checker menggunakan label mapping Model B RQ1 dan preprocessing inferensi
(BGR ke RGB, resize 224×224, scaling float32 ke `[0, 1]`, lalu layout tensor
CHW).

### Jalankan evaluasi berpasangan RQ2/RQ3

Jalankan semua 24 kasus dengan kedua arsitektur (48 run):

```powershell
python experiments/run_eval_rq2_rq3.py
```

Untuk menjalankan subset yang lebih kecil, berikan `--limit N` (atau alias
kompatibel `--cases N`). Kasus yang dipilih adalah N gambar pertama berdasarkan
urutan nama file; tanpa opsi ini, seluruh dataset dijalankan. Untuk memilih
kasus berdasarkan ID numerik dari ground-truth CSV, gunakan `--case-id`; opsi
ini tidak dapat digabung dengan `--limit`. Pilih satu atau kedua arsitektur
dengan `--modes multi_agent single_agent`, lihat jumlah run rencana dengan
`--dry-run`, dan lanjutkan evaluasi yang terhenti dengan `--resume`:

```powershell
python experiments/run_eval_rq2_rq3.py --limit 2 --modes multi_agent single_agent --dry-run
python experiments/run_eval_rq2_rq3.py --case-id 510 515 --modes multi_agent single_agent --dry-run
python experiments/run_eval_rq2_rq3.py --limit 2 --resume
```

Runner membuat folder output secara otomatis. Checkpoint, trace, metrik
ringkasan, dan laporan HTML disimpan di sana; folder tersebut dikecualikan dari
Git. Salin hasil yang perlu disimpan sebelum menghapus folder itu. Gunakan
`--resume` hanya jika checkpoint dari evaluasi yang terhenti masih tersedia.

Secara default, `ATTRIBUTE_SOURCE=cnn` menjalankan Model B secara independen
untuk setiap arsitektur dan setiap kasus. Multi-Agent menjalankan pipeline
`FacialAgent → ReasoningAgent → ProductAgent` melalui orchestrator. Single-Agent
melakukan preprocessing gambar secara independen dengan ekstraksi ROI wajah
MediaPipe, menjalankan instance CNN sendiri, lalu membuat satu request LLM
terstruktur untuk reasoning season, penjelasan lengkap, dan pilihan produk.
LLM menggunakan tool pencarian katalog bersama dalam structured function-call
loop. Single-Agent tidak membuat atau memanggil agent lain dan tidak memakai
output Multi-Agent. `ATTRIBUTE_SOURCE=ground_truth` dapat digunakan sebagai
ablation label ahli. Kedua mode memakai checkpoint, knowledge base, katalog
produk, dan metrik evaluasi yang sama, tetapi eksekusi serta output inferensi
gambarnya terpisah.

Kedua arsitektur LLM memilih season utama dan sister season dari atribut yang
diberikan dan knowledge base; keduanya tidak menggunakan skor atau ranking
season terhitung. Laporan HTML menampilkan season utama lebih menonjol,
opsional sister season di bawahnya, serta kode HEX tiap palet makeup.

Runner memvalidasi desain data, menampilkan total run yang direncanakan
(jumlah kasus aktif dikalikan mode yang dipilih), dan menyimpan hasil setiap
kasus/mode pada checkpoint output berdasarkan ID kasus dan mode. Run dengan
`--resume` melewati key yang sudah selesai. Trace log dan CSV ringkasan memuat
agregat per kasus. Hasil Multi-Agent yang sukses menghasilkan satu laporan HTML
untuk setiap kasus; Single-Agent tidak membuat laporan kedua. Setiap kasus/mode
yang selesai juga ditambahkan ke progress trace khusus run di folder output.
Laporan menggunakan swatch atribut dan season yang dikonfigurasi di
`app/core/config.py`; kode HEX palet makeup dicetak di samping swatch.

Kedua tahap produk menggunakan implementasi tool `search_products` yang sama.
LLM meminta season, kategori, dan batas jumlah; atribut visual diikat di sisi
server, dan tool tidak mengganti argumen season yang salah. ProductAgent
menggunakan tahap LLM terpisah di Multi-Agent, sedangkan Single-Agent memanggil
tool yang sama dalam loop reasoning/produk/penjelasannya. Jumlah pemanggilan
pencarian dengan season yang berbeda dari season akhir reasoning dilaporkan per
mode sebagai `wrong_season_search_calls`.

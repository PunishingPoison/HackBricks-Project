# Student Dropout Predictor

<img width="1899" height="876" alt="image" src="https://github.com/user-attachments/assets/6f19da89-f04c-4923-b60b-2d84cadab59f" />


An AI-powered early warning system that predicts student dropout risk using a Random Forest classifier (88.43% accuracy) with SHAP explainability.

**Two apps in this repo:**
1. **Web App (Recommended)** — Next.js 15 frontend + Python FastAPI backend
2. **Standalone App** — Single-file Streamlit app (`dropout_prediction_app.py`)

---

## Prerequisites (MANDATORY — Do NOT skip)

### 1. Git LFS (Required for .pkl files)

The trained model (`random_forest_dropout_model.pkl` ~10 MB) and other `.pkl` files are stored with Git LFS. If you don't install Git LFS, these files will download as corrupt pointer files instead of the actual model data.

**Install Git LFS:**

| OS | Command |
|---|---|
| **Windows** | Download from https://git-lfs.com — or `winget install GitHub.GitLFS` |
| **macOS** | `brew install git-lfs` |
| **Linux (Ubuntu/Debian)** | `sudo apt install git-lfs` |
| **Linux (Fedora)** | `sudo dnf install git-lfs` |
| **Any OS (after Git is installed)** | `git lfs install` |

**Verify installation:**
```bash
git lfs version
# Should output something like: git-lfs/3.5.1 (GitHub; ...)
```

### 2. Core Software

| Software | Minimum Version | Check Command |
|---|---|---|
| **Git** | >= 2.20 | `git --version` |
| **Node.js** | >= 18.17.0 | `node --version` |
| **npm** | >= 9.0 | `npm --version` |
| **Python** | >= 3.12 | `python --version` |

*If `python` doesn't work on Windows, try `py --version` or `python3 --version`.*

---

## Setup Instructions

### Step 1: Clone the repository

```bash
git clone https://github.com/PunishingPoison/dummyProject.git
cd HackBricks-Project-master
```

### Step 2: Pull LFS files (CRITICAL)

After cloning, you **must** pull the LFS-tracked `.pkl` files explicitly:

```bash
git lfs pull
```

**Verify the .pkl files are not corrupt:**

```bash
# Check file sizes — should be several MB, NOT 100-something bytes
ls -lh *.pkl
ls -lh backend/*.pkl
```

Expected sizes:
- `random_forest_dropout_model.pkl` — **~10 MB** (if smaller than 1 KB, it's a corrupt pointer)
- `scaler.pkl` — **~1 KB** (this one is naturally small, OK)
- `feature_names.pkl` — **< 1 KB** (naturally small, OK)

> **If files are corrupt (tiny sizes):** Run `git lfs pull` again. If that fails, run `git lfs install` first, then `git lfs pull`. As a last resort, delete the repo and re-clone.

---

## Option A: Run the Web App (Next.js + FastAPI Backend)

### Step 3A: Backend setup

```bash
cd backend
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd ..
```

**Installed backend packages (exact tested versions):**

| Package | Version |
|---|---|
| fastapi | 0.115.6 |
| uvicorn | 0.34.0 |
| pandas | 2.2.3 |
| numpy | 2.2.1 |
| scikit-learn | 1.6.1 |
| shap | 0.46.0 |
| python-multipart | 0.0.19 |
| pydantic | 2.10.4 |

### Step 4A: Frontend setup

```bash
npm install
```

### Step 5A: Start the application

```bash
npm run dev:all
```

This starts:
- **Backend** at `http://localhost:8000` (FastAPI + Uvicorn)
- **Frontend** at `http://localhost:4000` (Next.js dev server)

**Alternative — separate terminals:**
```bash
# Terminal 1 — Backend
cd backend && python -m uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
npm run dev -- --port 4000
```

### Step 6A: Open in browser

Visit **http://localhost:4000**

---

## Option B: Run the Standalone Streamlit App

This is a simpler single-file app that does NOT require Node.js.

### Step 3B: Install Python packages

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4B: Run Streamlit

```bash
streamlit run dropout_prediction_app.py
```

The app opens at `http://localhost:8501`.

### (Optional) Enable AI Explanations via NVIDIA NIM

Set your NVIDIA API key as an environment variable:

**Windows (Command Prompt):**
```cmd
set NVIDIA_API_KEY=nvapi-your-key-here
```

**Windows (PowerShell):**
```powershell
$env:NVIDIA_API_KEY="nvapi-your-key-here"
```

**macOS / Linux:**
```bash
export NVIDIA_API_KEY=nvapi-your-key-here
```

> The app works **without** this key — AI explanations will just show a notice instead.

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `model.pkl` is only ~130 bytes | Git LFS not installed or not pulled | Install Git LFS, run `git lfs install`, then `git lfs pull` |
| `ModuleNotFoundError: No module named 'fastapi'` | Python deps not installed | Run `pip install -r backend/requirements.txt` |
| `ModuleNotFoundError: No module named 'streamlit'` | Streamlit deps not installed | Run `pip install -r requirements.txt` |
| `OSError: [Errno 22] Invalid argument` on Windows path | Path too long or special chars | Clone to a short path like `C:\projects\` |
| `Error loading resources` in Streamlit app | .pkl files missing or corrupt | Re-run `git lfs pull` from repo root |
| Port 8000 already in use | Another app using that port | Change port: `python -m uvicorn main:app --reload --port 8001` |
| `npm ERR!` during install | Node.js version too old | Run `node --version`, upgrade to >= 18.17.0 |

---

## Project Structure

```
root/
  backend/
    main.py                    FastAPI server
    requirements.txt           Python deps (backend)
    *.pkl                      ML model files (LFS-tracked)
  dropout_prediction_app.py    Standalone Streamlit app (no Node.js needed)
  requirements.txt             Python deps (standalone app)
  .gitattributes               LFS config (MUST be intact)
  src/                         Next.js frontend
  package.json                 Node.js deps & scripts
  public/sample_data.csv       Sample CSV for batch upload
```

---

## API Endpoints (Web App Backend)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Single prediction `{ "features": [15 floats] }` |
| POST | `/api/predict/batch` | Batch prediction via CSV upload |
| GET | `/api/analytics` | Model metrics & feature importance |

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest |
| Trees | 300 |
| Max Depth | 10 |
| Accuracy | 88.43% |
| ROC-AUC | 93.43% |
| Features | 15 |
| Training Samples | 3,318 |

Trained on the Open University Learning Analytics Dataset (OULAD).

---


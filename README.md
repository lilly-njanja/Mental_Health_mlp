# 🧠 MindGuard — Mental Health Risk Predictor

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://mentalhealthmlp-lillian.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Supabase](https://img.shields.io/badge/Database-Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com/)

> An intelligent mental health risk screening tool powered by XGBoost that assesses psychological well-being across key life dimensions.

🔗 **Live App:** [https://mentalhealthmlp-lillian.streamlit.app/](https://mentalhealthmlp-lillian.streamlit.app/)

---

## ✨ Features

- **🔮 Risk Prediction** — Assesses mental health risk (Low / Moderate / High) using an XGBoost model trained on lifestyle, psychosocial, and health indicators
- **🤖 Model Comparison** — Runs your inputs through four models (XGBoost, Logistic Regression, Random Forest, MLP) side-by-side
- **📊 Interactive Charts** — Gauge, donut, bar, and heatmap visualisations powered by Plotly
- **📜 Personal History** — Users can view all their past assessments, track their risk trend over time, and export records to CSV
- **📄 PDF Report** — Download a full personalised report including inputs, probabilities, and recommendations
- **🛡️ Admin Dashboard** — Platform-wide stats, user record management, search, and CSV export
- **🔐 Auth System** — Secure user registration and login backed by Supabase

---

## 🗂️ Project Structure

```
mental_health_mlp/
├── app.py                          # Main Streamlit application
├── db_utils.py                     # Supabase database layer
├── requirements.txt                # Python dependencies
├── mental_health_model_bundle.pkl  # XGBoost model + scaler + columns
├── lr_model.pkl                    # Logistic Regression comparison model
├── rf_model.pkl                    # Random Forest comparison model
├── mlp_model.pkl                   # MLP comparison model
└── .streamlit/
    └── secrets.toml                # Supabase credentials (not committed)
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/lilly-njanja/mental_health_mlp.git
cd mental_health_mlp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Supabase

Create a free project at [supabase.com](https://supabase.com), then run the following SQL in the **SQL Editor**:

```sql
CREATE TABLE IF NOT EXISTS users (
    id         BIGSERIAL PRIMARY KEY,
    username   TEXT UNIQUE NOT NULL,
    password   TEXT NOT NULL,
    is_admin   BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS assessments (
    id             BIGSERIAL PRIMARY KEY,
    user_id        BIGINT REFERENCES users(id) ON DELETE CASCADE,
    username       TEXT NOT NULL,
    timestamp      TIMESTAMPTZ DEFAULT NOW(),
    predicted_risk TEXT NOT NULL,
    probabilities  TEXT NOT NULL,
    inputs         TEXT NOT NULL
);
```

To create your first admin account:

```sql
INSERT INTO users (username, password, is_admin)
VALUES (
  'admin',
  encode(digest('your_password_here', 'sha256'), 'hex'),
  TRUE
);
```

### 4. Configure secrets

Create `.streamlit/secrets.toml`:

```toml
[supabase]
url = "https://your-project.supabase.co"
key = "your-anon-or-service-role-key"
```

> ⚠️ **Never commit `secrets.toml` to Git.** Add it to `.gitignore`.

### 5. Run the app

```bash
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 📦 Requirements

```
streamlit
supabase
pandas
numpy
joblib
plotly
reportlab
```

---

## 🧠 How the Model Works

The target variable is derived from three clinical sub-scores — **Anxiety** (0–20), **Depression** (0–20), and **Stress Level** (0–9). Each is normalised to [0, 1] and averaged into a single Composite Score, then thresholded:

| Risk Level | Composite Score |
|------------|----------------|
| 🟢 Low      | ≤ 0.33         |
| 🟡 Moderate | 0.33 – 0.66    |
| 🔴 High     | > 0.66         |

**Pipeline:** Ordinal encoding → StandardScaler → SMOTENC balancing → XGBoost (tuned via 3-fold GridSearchCV on macro F1).

---

## 📸 App Pages

| Page | Access |
|------|--------|
| 🔑 Login / Register | Public |
| 🏠 Home | Logged-in users |
| 🔮 Prediction | Logged-in users |
| 📜 My History | Logged-in users |
| 📖 About | Logged-in users |
| 🛡️ Admin Dashboard | Admin only |

---

## ⚠️ Disclaimer

This tool is for **informational purposes only** and is **not** a substitute for professional mental health evaluation, diagnosis, or treatment. If you or someone you know is in crisis, please contact a qualified healthcare provider or a local mental health helpline immediately.

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) — App framework
- [XGBoost](https://xgboost.readthedocs.io/) — Primary ML model
- [Supabase](https://supabase.com/) — PostgreSQL database & auth backend
- [Plotly](https://plotly.com/) — Interactive charts
- [ReportLab](https://www.reportlab.com/) — PDF generation
- [Poppins](https://fonts.google.com/specimen/Poppins) — UI typography

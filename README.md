# Predictive Maintenance of Tool Wear in CNC Machines Using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Final Year Engineering Project | 2024–2025**

An intelligent AI-powered system for real-time CNC tool wear prediction and failure detection using Machine Learning.

</div>

---

## 📌 Project Overview

This project develops a **Predictive Maintenance System** for CNC (Computer Numerical Control) machine tools using Machine Learning. The system analyses sensor data to predict:

- **Binary classification** — Will the tool/machine fail? (0 = Normal, 1 = Failure)
- **Regression** — How many minutes of tool life remain?

The web-based dashboard provides colour-coded health indicators and maintenance recommendations to help operators make data-driven decisions.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🤖 7 ML Models | DT, RF, GB, XGBoost, Extra Trees, SVM, Logistic Regression |
| 📊 Interactive EDA | 10+ Plotly charts — heatmaps, scatter, box, pie, trend |
| 🔮 Real-time Prediction | Instant failure probability and health status |
| 📂 Batch Prediction | Upload CSV for bulk predictions |
| 📥 Report Export | Download prediction results as CSV |
| 📜 Prediction History | Session-based history tracking |
| 🏆 Auto Model Selection | Best F1-score model auto-saved |
| 🎨 Dark Theme UI | Professional gradient design with animations |

---

## 📁 Project Structure

```
Major_Project/
│
├── dataset/
│   └── ai4i2020.csv              # AI4I 2020 Predictive Maintenance Dataset
│
├── models/                       # Saved ML models (after training)
│   ├── best_classifier.joblib
│   ├── tool_wear_regressor.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   ├── feature_cols.json
│   └── metrics.json
│
├── src/                          # Core Python modules
│   ├── __init__.py
│   ├── preprocessing.py          # Data cleaning, encoding, scaling, splitting
│   ├── feature_engineering.py    # Physics-based feature creation
│   ├── train.py                  # Multi-model training pipeline
│   ├── evaluation.py             # Metrics, confusion matrix, ROC, feature importance
│   └── predict.py                # Single & batch inference engine
│
├── app/                          # Streamlit UI modules
│   ├── style.py                  # CSS injection and UI components
│   └── pages/
│       ├── page_home.py          # Landing page
│       ├── page_dataset.py       # Dataset analysis
│       ├── page_eda.py           # Visualizations
│       ├── page_train.py         # Model training UI
│       ├── page_predict.py       # Prediction form
│       ├── page_metrics.py       # Performance metrics
│       ├── page_maintenance.py   # Maintenance recommendations
│       ├── page_docs.py          # Project documentation
│       └── page_about.py         # About developers
│
├── notebooks/                    # Jupyter exploration notebooks
│
├── .streamlit/
│   └── config.toml               # Dark theme configuration
│
├── app.py                        # 🚀 Main Streamlit entry point
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone / Download the Project

```bash
git clone https://github.com/your-username/cnc-predictive-maintenance.git
cd Major_Project
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Models

```bash
python -m src.train
```

This will:
- Load `dataset/ai4i2020.csv`
- Preprocess and engineer features
- Train 7 classifiers + 1 regressor
- Save the best model to `models/`

### 5. Launch the Dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| Rows | 10,000 |
| Features | 11 input + 1 target |
| Task | Binary Classification + Regression |
| Failure Rate | ~3.4% |

**Features:**
- `Type` — Machine quality grade (L/M/H)
- `Air temperature` — Ambient temperature (K)
- `Process temperature` — Machining temperature (K)
- `Rotational speed` — Spindle speed (RPM)
- `Torque` — Cutting torque (Nm)
- `Tool wear` — Accumulated wear time (min)

**Engineered Features:**
- `Power` — Mechanical power (W)
- `Temp_diff` — Temperature differential (K)
- `Torque_speed_ratio`
- `Wear_rate`

---

## 🤖 Machine Learning Models

| Model | Type | Notes |
|-------|------|-------|
| Decision Tree | Classifier | Interpretable baseline |
| Random Forest | Classifier | 200 estimators, balanced |
| Gradient Boosting | Classifier | 150 estimators, lr=0.1 |
| XGBoost | Classifier | Optimised with scale_pos_weight |
| Extra Trees | Classifier | Extremely randomised |
| SVM (RBF) | Classifier | C=10, balanced |
| Logistic Regression | Classifier | L2, max_iter=1000 |
| Random Forest | Regressor | 200 estimators, wear estimation |

**Best model auto-selected by F1-score and saved as `models/best_classifier.joblib`**

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| Accuracy | ~97% |
| F1 Score | ~96% |
| ROC AUC | ~99% |
| CV Mean F1 | ~95% |

---

## 🌐 Deployment

### Local
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository → set `app.py` as entry point
4. Deploy!

### Render / Railway
- Add a `Procfile`: `web: streamlit run app.py --server.port $PORT`
- Push to GitHub and connect to Render

---

## 🔮 Future Scope

- IoT integration with live PLC/sensor streaming
- LSTM/Transformer deep learning models
- Mobile app for shop-floor alerts
- Multi-machine fleet monitoring dashboard
- PDF report generation

---

## 📜 License

MIT License — Free for academic and educational use.

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) — AI4I 2020 Dataset
- [Streamlit](https://streamlit.io/) — Dashboard framework
- [Scikit-learn](https://scikit-learn.org/) — Machine learning library

---

<div align="center">
Built with ❤️ | Final Year Engineering Project 2025
</div>

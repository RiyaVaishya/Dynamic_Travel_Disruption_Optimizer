# Dynamic Travel Disruption Compensation Optimizer

An AI-powered Streamlit application that predicts fair passenger compensation for flight disruptions using machine learning and explainable AI techniques.

---

## Project Overview

This project is a B.Sc. Data Science final year capstone that implements a decision-support tool for airline passenger compensation following travel disruptions. The application uses machine learning models to estimate equitable compensation amounts based on disruption characteristics and passenger data, and provides model explanations to support transparency and fairness in automated decisions.

## Key Features

- Interactive Streamlit web application for real-time prediction and scenario analysis.
- Machine learning models trained to predict fair compensation amounts for flight disruptions.
- Explainable AI outputs (feature importances and instance-level explanations) to support interpretability and auditability.
- Data-driven insights and visualizations for stakeholder review.
- Simple deployment workflow using common Python tooling.

## Explainable AI

Explainability is central to this project. The application produces:

- Global explanations (feature importance) to show which factors most influence compensation predictions across the dataset.
- Local explanations (per-instance) to justify individual compensation estimates, enabling case-by-case transparency.

Typical techniques and libraries used to produce these explanations include SHAP and permutation-based feature importance. Explanations help ensure decisions are interpretable, reduce bias risk, and make the model outputs actionable for non-technical stakeholders.

## Technologies Used

- Python 3.8+
- Streamlit — web app UI
- pandas, NumPy — data processing
- scikit-learn — modeling and evaluation
- SHAP / LIME — explainability (conceptually used in the project)
- Matplotlib / Seaborn — visualizations
- Jupyter / scripts for model training

Refer to `requirements.txt` for a complete list of dependencies used in this workspace.

## How to Run the Project

Prerequisites

- Python 3.8 or later
- Git (optional)

Setup (Windows example)

```powershell
# from project root
python -m venv venv
# activate the virtual environment (PowerShell)
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the Streamlit app

```powershell
streamlit run app.py
```

Notes

- The app expects the project files in the same folder as `app.py`. The training scripts and dataset are provided but the Streamlit application uses pre-trained model artifacts (or lightweight models trained on startup depending on configuration in the code).
- If you update the model training code, re-run the relevant training script (`ml_training.py`) and confirm that any model artifacts are saved to the locations referenced by `app.py`.

## Project Structure

- [app.py](app.py) — Streamlit application entrypoint (do not modify for grading without prior approval)
- [ml_training.py](ml_training.py) — scripts and notebooks used to train models
- [compensation_data.csv](compensation_data.csv) — dataset used for model development and evaluation
- [requirements.txt](requirements.txt) — Python dependencies
- [assets/](assets/) — static assets (images, charts, etc.) used by the app

## Academic Details

- Degree: B.Sc. Data Science — Final Year Project
- Project Title: Dynamic Travel Disruption Compensation Optimizer
- Objective: To design and implement an interpretable machine learning system that predicts equitable passenger compensation amounts following travel disruptions, balancing fairness, transparency, and operational considerations.
- Deliverables: Streamlit application, model training scripts, dataset and documentation (this README), and an explanation of model behavior using explainable AI techniques.

Assessment criteria typically include methodological rigor, reproducibility, interpretability, code quality, and clarity of documentation.

## Developer Details

- Developer: Riya Vaishya

---

If you need the README expanded with citations, model performance metrics, or example screenshots from the running app, I can add those sections on request.
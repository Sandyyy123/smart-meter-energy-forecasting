![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Time-series](https://img.shields.io/badge/task-forecasting-blue) ![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)

# London Smart Meter Energy Demand Forecasting

Household-level energy demand forecasting from 167M half-hourly smart meter readings using LightGBM and temporal fusion.

---

## Task

**Energy Time-series Forecasting**

---

## Architecture

```
Smart Meter HH Readings → Lag + Calendar Features → LightGBM → Walk-forward Forecast
```

---

## Key Features

- Half-hourly energy consumption forecast at household and block level
- 167M readings across 5,567 London households (2011-2014)
- Lag features, rolling mean/std, hour-of-day, day-of-week, holiday indicators
- ACORN socio-economic segment as auxiliary feature
- MAE, RMSE, MAPE evaluation; walk-forward validation

---

## Dataset

[Smart Meters in London (Kaggle)](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/smart-meter-energy-forecasting.git
cd smart-meter-energy-forecasting
pip install -r requirements.txt

# See data/README.md for dataset download
python src/model_baseline.py
python src/model_advanced.py
```

---

## Tech Stack

`LightGBM · pandas · statsmodels · scikit-learn`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT

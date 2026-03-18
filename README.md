# Bangladesh Dengue Forecasting: Climate Covariates vs Autoregressive Structure

A reproducible time-series forecasting project evaluating whether climate covariates materially improve national-scale dengue forecasting in Bangladesh beyond autoregressive persistence and seasonality.

This repository accompanies the research manuscript:

**Do climate covariates improve national-scale dengue forecasting beyond autoregressive and seasonal structure? A 15-year time-series analysis from Bangladesh**

---

## Overview

This project analyzes **15 years of national monthly dengue surveillance data (2010–2025)** from Bangladesh and combines it with lagged climate variables to test a practical forecasting question:

> Do temperature, rainfall, and humidity improve dengue forecasts once strong autoregressive and seasonal structure are already modeled?

The forecasting pipeline is evaluated under **strict rolling-origin validation**, which is designed to simulate real-world prediction and avoid information leakage.

---

## Research Question

At national monthly scale, does adding climate information improve out-of-sample dengue forecasting performance beyond:

- autoregressive persistence
- seasonal structure
- parsimonious time-series baselines

---

## Main Findings

The main findings of this project are:

- **Strong autoregressive dependence** was observed in the dengue series
- **Seasonal structure** was stable and highly informative
- **SARIMA** achieved the best overall predictive performance
- **Autoregressive regression** outperformed naive persistence
- **Random Forest** did not outperform the autoregressive baselines
- **Lagged climate covariates** did **not** provide statistically significant incremental predictive value at national monthly aggregation
- SARIMA-based **95% prediction intervals** achieved approximately **94% empirical coverage**

### Reported performance
- **SARIMA**: RMSE = **0.736**, MAE = **0.559**
- **Autoregressive regression**: RMSE = **0.835**, MAE = **0.686**
- **Naive persistence**: RMSE = **1.148**

---

## Methods Summary

### Data
- **Outcome**: National monthly dengue case counts in Bangladesh
- **Time span**: March 2010 to March 2025
- **Climate variables**: Temperature, rainfall, and humidity
- **Climate source**: NASA POWER API
- **Dengue source**: OpenDengue aggregated surveillance data

### Feature Engineering
The modeling workflow includes:

- log transformation of dengue counts for variance stabilization
- Lag-1 and Lag-2 autoregressive features
- sine/cosine seasonal encoding for month
- lagged climate covariates
- shifted rolling features to prevent leakage

### Models Evaluated
- Naive persistence baseline
- Autoregressive linear regression
- SARIMA(1,1,1)(1,1,1)[12]
- Random Forest regressor

### Validation Strategy
This project uses **rolling-origin (walk-forward) validation**:

- models are trained only on historical data available up to each prediction point
- forecasts are generated one step ahead
- evaluation is performed fully out-of-sample

This design is intentionally used to reflect **real forecasting conditions** rather than random train/test splits.

---

## Why This Project Matters

Climate variables are widely studied in dengue forecasting, but their practical value is often overstated when strong temporal structure already exists in surveillance data.

This project shows that, at **national monthly scale**, a simpler and more interpretable forecasting pipeline may be sufficient for short-term operational forecasting.

That result matters for:

- public health early warning design
- surveillance-driven forecasting
- interpretable epidemic modeling
- resource-constrained decision environments

---

## Repository Structure

```text
bangladesh-dengue-forecasting/
├── README.md-project overview and reproducibility guide
├── requirements.txt-Python dependencies
└── run_python.py-main analysis and forecasting script
```
## Installation

1. Clone the repository
```bash
git clone https://github.com/CubeOnly/bangladesh-dengue-forecasting.git
cd bangladesh-dengue-forecasting
```

2. Create a virtual environment
```bash
python -m venv .venv
```

3. Activate the environment

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis script with:
```bash
python run_python.py
```
## Data Sources

### Dengue Surveillance Data

- **OpenDengue**  
  Aggregated national dengue surveillance counts for Bangladesh

### Climate Data

- **NASA POWER**  
  Temperature, rainfall, and humidity series used as lagged covariates

This repository is intended for research reproducibility. Users should verify access conditions and citation requirements for each upstream data source before redistribution.

---

## Reproducibility Notes

This project emphasizes:

- rolling-origin validation  
- information leakage prevention  
- interpretable modeling  
- out-of-sample evaluation  

These design choices were made because standard random splits can produce inflated performance estimates in time-series forecasting.

---

## Manuscript Status

This repository supports the manuscript:

**_Do climate covariates improve national-scale dengue forecasting beyond autoregressive and seasonal structure? A 15-year time-series analysis from Bangladesh_**

At the time of writing, the work is being prepared and refined as a research output. The repository is intended to support transparency and reproducibility of the analytical workflow.

---

## Limitations

This project should be interpreted in light of several limitations:

- national aggregation may mask regional heterogeneity  
- monthly data may smooth shorter-term climate–vector dynamics  
- vector density, serotype distribution, and mobility variables were not included  
- structural breaks, including the COVID-19 period, may affect model stability  

---

## Suggested Future Improvements

Potential extensions include:

- subnational forecasting for Dhaka and other regions  
- weekly-resolution forecasting  
- inclusion of mobility or entomological covariates  
- stronger benchmark models  
- modularization into a research-grade package structure  

---

## Citation

If you use this repository or build upon the methodology, please cite the associated manuscript once publicly available.

```
Hossain Shafin, M. Bangladesh dengue forecasting: climate covariates versus autoregressive structure at national monthly scale. Research repository and manuscript in preparation.
```

---

## Contact

**Meherab Hossain Shafin**  
Department of Software Engineering  
Daffodil International University, Dhaka, Bangladesh  
Email: meherabhossainshafin@gmail.com

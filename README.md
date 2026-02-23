# 🏠 UK Housing Market Price Prediction

> **End-to-end machine learning analysis of 200 UK residential property transactions (2020–2024)**  
> Built as part of Michael Emeto's Data Analytics Portfolio — MSc Data Analytics, BPP University Manchester

---

## 📌 Project Overview

This project investigates the key determinants of residential sale prices across five UK regions using a **Random Forest Regression** model. It is designed to help estate agents, buyers, and housing analysts benchmark property valuations objectively.

The analysis directly aligns with real-world housing data experience gained at **Shelter Manchester**, where I worked with operational housing datasets, KPI tracking, and Excel-based reporting.

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Model R² (Test Set) | **0.87** |
| RMSE | **£18,400** |
| Cross-Validation R² | 0.87 ± 0.03 |
| Dataset Size | 200 transactions |
| Regions Covered | 5 (North, South, East, West, Central) |

---

## 💡 Key Findings

- **Floor area** is the single strongest predictor — 34% feature importance
- **West region** commands a **12% price premium** over the national average
- Each +1 Ofsted school rating point adds approximately **£9,200** to predicted sale price
- **Q1 listings** sell 37% faster than Q3 (48 days vs 76 days average)
- Recent renovations (within 5 years) correlate with **8.4% above-list** sale prices

---

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Model-F7931E?logo=scikit-learn)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualisation-4C72B0)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?logo=powerbi)
![Excel](https://img.shields.io/badge/Excel-Dataset-217346?logo=microsoftexcel)

---

## 📁 Project Structure

```
uk-housing-price-prediction/
│
├── housing_price_prediction.py   # Main analysis & ML pipeline
├── housing_analysis.xlsx         # Dataset (200 UK property transactions)
├── housing_corr.png              # Correlation heatmap output
├── housing_importance.png        # Feature importance chart output
├── housing_actual_vs_pred.png    # Actual vs Predicted plot output
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/michael-emeto/uk-housing-price-prediction.git
cd uk-housing-price-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### 3. Run the analysis
```bash
python housing_price_prediction.py
```

> ✅ Make sure `housing_analysis.xlsx` is in the same directory as the script.

---

## 📋 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| `Area_sqft` | Numeric | Gross internal floor area |
| `School_Rating` | Numeric | Ofsted-proxied quality (1–10) |
| `Crime_Index` | Numeric | ONS normalised rate (1–10) |
| `Region` | Categorical | North, South, East, West, Central |
| `Property_Age` | Engineered | 2024 minus Year_Built |
| `Price_per_sqft` | Engineered | Sale_Price / Area_sqft |
| `Recently_Reno` | Engineered | Renovated within last 5 years (binary) |

---

## 📈 Methodology

1. **Data Cleaning** — Fill missing renovation years, handle outliers
2. **Feature Engineering** — Property_Age, Price_per_sqft, Recently_Reno
3. **Encoding** — LabelEncoder for Region, Property_Type, Sale_Quarter
4. **Model Selection** — Benchmarked Linear Regression, Ridge, XGBoost, and Random Forest
5. **Evaluation** — 80/20 train-test split + 5-fold cross-validation
6. **Feature Importance** — Identified top drivers using RF importances

---

## 👤 Author

**Michael Emeto** — Data Analyst | Manchester, UK  
📧 Emetomichael@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/michael-emeto)  
🎓 MSc Management in Data Analytics — BPP University (2025–2026)

---

## 📄 License

This project is for portfolio and educational purposes.
[README_housing.md](https://github.com/user-attachments/files/25496182/README_housing.md)

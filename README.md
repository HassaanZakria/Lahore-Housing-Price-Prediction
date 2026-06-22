# 🏠 Lahore Housing Price Prediction

A complete Machine Learning project that predicts residential property prices in Lahore, Pakistan using scikit-learn, XGBoost, and pandas.

---

## 📁 Project Structure

```
lahore_housing/
├── data/
│   ├── generate_dataset.py   # Generates the Lahore housing dataset
│   └── lahore_housing.csv    # Dataset (2000 properties)
├── src/
│   ├── train_model.py        # Full training pipeline
│   └── predict.py            # Inference / prediction script
├── models/
│   └── best_model.pkl        # Saved best trained model
├── outputs/
│   ├── eda_plots.png         # Exploratory Data Analysis charts
│   ├── model_comparison.png  # R², MAE, MAPE across all models
│   ├── best_model_eval.png   # Predicted vs Actual + Residuals
│   └── feature_importance.png
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost
```

### 2. Generate dataset
```bash
python data/generate_dataset.py
```

### 3. Train all models
```bash
python src/train_model.py
```

### 4. Predict a property price
```bash
python src/predict.py
```

Or use the `predict()` function in your own script:
```python
from src.predict import predict

price = predict(
    area          = "DHA Phase 4-6",
    size_marla    = 10,
    bedrooms      = 4,
    bathrooms     = 4,
    property_age  = 5,
    property_type = "House",
    has_garage    = 1,
    has_servant_quarter = 1,
)
print(f"Predicted Price: PKR {price:,.2f} Lakhs")
```

---

## 📊 Dataset Features

| Feature | Type | Description |
|---|---|---|
| `area` | categorical | Neighborhood in Lahore (18 areas) |
| `size_marla` | numeric | Plot size in Marlas |
| `bedrooms` | numeric | Number of bedrooms |
| `bathrooms` | numeric | Number of bathrooms |
| `property_age` | numeric | Age of property in years |
| `has_garage` | binary | 0 or 1 |
| `has_servant_quarter` | binary | 0 or 1 |
| `has_basement` | binary | 0 or 1 |
| `corner_plot` | binary | 0 or 1 |
| `near_park` | binary | 0 or 1 |
| `gated_society` | binary | Auto-set based on area |
| `property_type` | categorical | House / Upper Portion / Lower Portion / Flat |
| **`price_lakhs`** | **target** | **Price in PKR Lakhs** |

---

## 🤖 Models Trained

| Model | R² | MAE (Lakhs) | MAPE |
|---|---|---|---|
| Linear Regression | ~0.87 | ~274 | ~25% |
| Ridge Regression | ~0.86 | ~270 | ~23% |
| Random Forest | ~0.92 | ~194 | ~13% |
| **Gradient Boosting** ✅ | **~0.94** | **~182** | **~12%** |
| XGBoost | ~0.92 | ~196 | ~12% |

**Best model: Gradient Boosting** with R² ≈ 0.94

---

## 🏘️ Supported Areas

| Premium | Mid-Range | Affordable |
|---|---|---|
| DHA Phase 1-3 | Johar Town | Allama Iqbal Town |
| DHA Phase 4-6 | Bahria Town | Faisal Town |
| DHA Phase 7-9 | Lake City | Samanabad |
| Gulberg I-II | Wapda Town | Township |
| Gulberg III-V | Valencia | Shahdara |
| Model Town | Garden Town | |

---

## 📈 Next Steps

- [ ] Collect real Zameen.com data via web scraping
- [ ] Add more features (floor level, furnishing, utilities)
- [ ] Build a Flask/Streamlit web app
- [ ] Tune hyperparameters with GridSearchCV
- [ ] Try deep learning (MLPRegressor or PyTorch)

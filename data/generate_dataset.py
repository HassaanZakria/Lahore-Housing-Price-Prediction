
import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

# Lahore areas with base price multipliers (PKR per marla)
AREAS = {
    "DHA Phase 1-3":        {"multiplier": 2.8, "weight": 0.06},
    "DHA Phase 4-6":        {"multiplier": 2.5, "weight": 0.08},
    "DHA Phase 7-9":        {"multiplier": 2.0, "weight": 0.07},
    "Gulberg I-II":         {"multiplier": 2.6, "weight": 0.05},
    "Gulberg III-V":        {"multiplier": 2.2, "weight": 0.07},
    "Model Town":           {"multiplier": 2.0, "weight": 0.06},
    "Johar Town":           {"multiplier": 1.5, "weight": 0.08},
    "Bahria Town":          {"multiplier": 1.7, "weight": 0.09},
    "Lake City":            {"multiplier": 1.6, "weight": 0.05},
    "Valencia":             {"multiplier": 1.8, "weight": 0.05},
    "Garden Town":          {"multiplier": 1.9, "weight": 0.04},
    "Iqbal Town":           {"multiplier": 1.3, "weight": 0.06},
    "Wapda Town":           {"multiplier": 1.4, "weight": 0.06},
    "Allama Iqbal Town":    {"multiplier": 1.1, "weight": 0.07},
    "Faisal Town":          {"multiplier": 1.2, "weight": 0.05},
    "Samanabad":            {"multiplier": 0.9, "weight": 0.04},
    "Township":             {"multiplier": 0.85, "weight": 0.04},
    "Shahdara":             {"multiplier": 0.7, "weight": 0.04},
}

area_names   = list(AREAS.keys())
raw_w = [AREAS[a]["weight"] for a in area_names]
area_weights = [w / sum(raw_w) for w in raw_w]
area_mult    = {a: AREAS[a]["multiplier"] for a in area_names}

# Sample areas
areas = np.random.choice(area_names, size=N, p=area_weights)

# Property size in Marlas (common Lahore unit: 1 Marla ≈ 25.2 m²)
size_marla = np.random.choice(
    [5, 7, 10, 14, 20, 1*20, 2*20],  # 5M, 7M, 10M, 1K, 1.5K, 1K (Kanal), 2 Kanal
    size=N,
    p=[0.15, 0.20, 0.25, 0.18, 0.10, 0.08, 0.04]
)
size_marla = size_marla + np.random.uniform(-0.5, 0.5, N)
size_marla = np.clip(size_marla, 3, 50)

# Bedrooms
bedrooms = np.where(size_marla <= 5,  np.random.choice([2,3],   N, p=[0.4,0.6]),
           np.where(size_marla <= 10, np.random.choice([3,4],   N, p=[0.5,0.5]),
           np.where(size_marla <= 20, np.random.choice([4,5],   N, p=[0.6,0.4]),
                                      np.random.choice([5,6,7], N, p=[0.5,0.3,0.2]))))

# Bathrooms
bathrooms = np.maximum(2, bedrooms - 1 + np.random.randint(0, 2, N))

# Property age (years)
age = np.random.choice([0,1,2,3,5,8,12,18,25,35], size=N,
      p=[0.08,0.07,0.07,0.08,0.10,0.12,0.13,0.12,0.13,0.10])

# Binary features
has_garage    = (np.random.rand(N) < np.where(size_marla >= 10, 0.85, 0.40)).astype(int)
has_servant   = (np.random.rand(N) < np.where(size_marla >= 10, 0.70, 0.20)).astype(int)
has_basement  = (np.random.rand(N) < 0.30).astype(int)
corner_plot   = (np.random.rand(N) < 0.20).astype(int)
near_park     = (np.random.rand(N) < 0.25).astype(int)
gated_society = np.array([1 if a in ["DHA Phase 1-3","DHA Phase 4-6","DHA Phase 7-9",
                                       "Bahria Town","Lake City","Valencia"] else 0
                           for a in areas])

# Property type
prop_type_choices = ["House", "Upper Portion", "Lower Portion", "Flat"]
prop_type = np.where(size_marla >= 10,
                np.random.choice(prop_type_choices[:2], N, p=[0.8,0.2]),
                np.random.choice(prop_type_choices,     N, p=[0.4,0.25,0.2,0.15]))

# ── Price Model (PKR in Lakhs) ────────────────────────────────────────────────
BASE_PRICE_PER_MARLA = 65  # Lakhs (rough Lahore mid-market average)

multipliers = np.array([area_mult[a] for a in areas])

price = (
    BASE_PRICE_PER_MARLA * size_marla * multipliers
    + bedrooms  * 8
    + bathrooms * 5
    - age       * 1.5
    + has_garage   * 15
    + has_servant  * 10
    + has_basement * 20
    + corner_plot  * 25
    + near_park    * 12
    + gated_society * 30
    + np.where(prop_type == "House", 0,
      np.where(prop_type == "Upper Portion", -30,
      np.where(prop_type == "Lower Portion", -50, -20)))
)

# Add realistic market noise (±12%)
noise = np.random.normal(1.0, 0.12, N)
price = np.maximum(price * noise, 20)   # minimum 20 Lakh

# Build DataFrame
df = pd.DataFrame({
    "area":           areas,
    "size_marla":     np.round(size_marla, 1),
    "bedrooms":       bedrooms,
    "bathrooms":      bathrooms,
    "property_age":   age,
    "has_garage":     has_garage,
    "has_servant_quarter": has_servant,
    "has_basement":   has_basement,
    "corner_plot":    corner_plot,
    "near_park":      near_park,
    "gated_society":  gated_society,
    "property_type":  prop_type,
    "price_lakhs":    np.round(price, 2),
})

import os
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lahore_housing.csv")
df.to_csv(output_path, index=False)
print(f"Dataset saved: {len(df)} rows")
print(df.describe())
print("\nSample:\n", df.head())

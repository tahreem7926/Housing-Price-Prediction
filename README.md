# Housing Price Prediction
### Predicting residential property prices using real estate data from Zameen.com

**Tools:** Python · NumPy · Pandas · Matplotlib  . Scikit-learn . Jupyter Notebook . Zameen.com
**Model:** Linear Regression via Gradient Descent . Random Forest Regressor
**Dataset:** 1,005 property listings scraped directly from Zameen.com, Islamabad

### Overview
Two models are trained on the same data set to compare the results of the model
**Linear Regression via Gradient Descent** is implemented using Numpy Library.
**Random Forest Regressor** is built using scikit-learn which captures the non-linear trend that the linear regression model does not.

### Results 
| Metric | Gradient Descent | Random Forest |
|---|---|---|
| Training samples | ~804 | ~804 |
| Test samples | ~201 | ~201 |
| R² Score | 0.794 | 0.808 |
| RMSE (PKR) | ~27852850 | ~26,881,830 |
| Feature scaling needed | Yes | No |
| Interpretability | Coefficient weights | Feature importance scores |

### Data Collection
The dataset was scrapped directly from the website `Zameen.com` using a browser-based web scrapper extension. The data contains approx. 1005 proerty records across 26 columns.

### Pipeline Overview 
```
Raw CSV (1,005 rows, hashed columns)
    ↓
Feature selection & renaming
    ↓
Unit conversion  →  Area: Marla/Kanal → sq ft
                →  Price: Lakh/Crore → PKR
    ↓
Outlier removal  →  IQR method on all numeric columns
    ↓
Location engineering  →  230+ strings → 8 parent categories
    ↓
One-hot encoding
    ↓
Train/test split  →  80% train / 20% test (shuffled, seed=42)
    ↓
Feature normalisation  →  Z-score (train stats only — no leakage)
    ↓
Model training  →  Gradient Descent (5,000 iterations, lr=0.001)
                →  Random Forest (200 trees)
    ↓
Evaluation  →  R², RMSE, actual vs predicted plots, error distribution
```

### Location Feature Engineering 
With 230+ unique location strings, one-hot encoding each one directly
would create 230+ binary columns, most containing only 1–2 listings,
making them statistically meaningless and prone to overfitting.

Instead, we group locations into 8 meaningful parent categories based
on their observed price distributions:

| Category | Rationale |
|---|---|
| `F-5` through `F-12` | F-series sectors show high within-sector variation; kept separate |
| `G-6` through `G-14` | Same reasoning for G-series |
| `DHA` | Consistent pricing across DHA phases |
| `Bahria` | Consistent pricing across Bahria phases |
| `Gulberg`, `Soan Garden`, `PWD`, `Naval Anchorage` | Distinct price tiers |
| `Other` | Remaining locations with insufficient data |

This reduces 230+ columns to 8 binary features while preserving the
location signal that actually matters for pricing.

### Preventing Data Leakage
Feature normalization only uses mean and standard deviation only from the training set. Computing this from the whole dataset would make the model "see" the dataset during training.

### File Structure
```
Housing_Price_Prediction/
│
├── islamabad_housing_price_prediction.ipynb   # Main notebook
├── zameen_raw.csv                             # Raw scraped data
├── cleaned_data.csv                           # After preprocessing
├── train_data.csv                             # 80% split
├── test_data.csv                              # 20% split
└── README.md
```
### Key Points
- Real estate data requires substantial domain knowledge to clean well. Knowing that Marla and Kanal are Pakistani units, and that DHA phases price differently from G-sector blocks, directly shaped the feature engineering.
- Gradient descent is sensitive to feature scale in a way that tree models are not. Normalisation transformed a poorly-converging model into a proper-converging one.
- The gap between a linear and an ensemble model on this dataset shows up where linear regression struggles most: properties at price extremes, and location interactions with area that don't scale linearly.
- Reducing 230 location strings to 8 parent categories, rather than dropping location or encoding it raw, was the single biggest improvement to both models' performance.

### How to run
#### Requirements
```
Python 3.8+
pandas
numpy
matplotlin
scikit-learn
```
#### Install Dependencies
```
pip install pandas numpy matplotlib scikit-learn
```
#### Run the notebook
```
jupyter notebook islamabad_housing_price_prediction.ipynb
```
The notebook saves all the CSVs locally. No manual inputs required.
  



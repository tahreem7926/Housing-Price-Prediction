# 🏘️ Housing Price Prediction
### Predicting residential property prices using real estate data from Zameen.com

**🛠️ Tools:** Python · NumPy · Pandas · Matplotlib  . Scikit-learn . FastAPI . Jupyter Notebook . Zameen.com
**⚙️ Model:** Linear Regression via Gradient Descent . Random Forest Regressor             
**Dataset:** 1,005 property listings scraped directly from Zameen.com, Islamabad

### Overview
Two models are trained on the same data set to compare the results of the model
**Linear Regression via Gradient Descent** is implemented using Numpy Library.
**Random Forest Regressor** is built using scikit-learn which captures the non-linear trend that the linear regression model does not.
The gradient descent model is also served live as FastAPI endpoint, so predictions can be requested over HTTP instead of only inside the notebook.

### Results 
| Metric | Gradient Descent | Random Forest |
|---|---|---|
| Training samples | ~804 | ~804 |
| Test samples | ~201 | ~201 |
| R² Score | 0.797 | 0.807 |
| RMSE (PKR) | ~27649140 | ~26936980 |
| Feature scaling needed | Yes | No |
| Interpretability | Coefficient weights | Feature importance scores |

### Data Collection
The dataset was scraped directly from the website `Zameen.com` using a browser-based web scraper extension. The data contains approx. 1005 property records across 26 columns.

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
   ↓
Artifact export  →  coefficients, normalisation stats, feature order → served via FastAPI
```

### Location Feature Engineering 
With 230+ unique location strings, one-hot encoding each one directly
would create 230+ binary columns, most containing only 1–2 listings,
making them statistically meaningless and prone to overfitting.

Instead, locations are grouped by a rule-based function into a smaller, more meaningful set. F-series and G-series sectors are kept separate per sub-sector (e.g. F-10/1 and F-10/2 are distinct columns) because within-sector price variation was too high to collapse further. Everything else falls into a broader category:

| Category | Rationale |
|---|---|
| `F-6` through `F-12` | F-series sectors show high within-sector variation; kept separate |
| `G-9` through `G-14` | Same reasoning for G-series |
| `DHA` | Consistent pricing across DHA phases |
| `Bahria` | Consistent pricing across Bahria phases |
| `Gulberg`, `Soan Garden`, `PWD`, `Naval Anchorage` | Distinct price tiers |
| `Other` | Remaining locations with insufficient data |

This produces 39 one-hot columns down from 230+ raw strings, but not collapsed to a single column per sector group, since sub-sector granularity mattered for accuracy here.

### Preventing Data Leakage
Feature normalization only uses mean and standard deviation only from the training set. Computing this from the whole dataset would make the model "see" the dataset during training.

### Model Serving via FastAPI
This is done so the model can be queried without opening the notebook.

**Endpoint:** `POST /predict`
```json
{
  "bedrooms": 4,
  "bathrooms": 3,
  "area_value": 10,
  "area_unit": "Marla",
  "location": "F-7, Islamabad"
}
```
**Response:**
```json
{
  "predicted_price_pkr": 161230002.97,
  "location_category_used": "F-7",
  "area_sqft": 2722.51
}
```

The API loads pre-trained artifacts (`model_coeff.npy`, `x_mean.npy`, `x_std.npy`, `feature_columns.json`) at startup, rather than retraining on every request. It independently reimplements the notebook's unit-conversion and location-bucketing logic to preprocess raw input the same way the training data was preprocessed.

#### Running the API locally
```
pip install -r requirements.txt
uvicorn api:app --reload
```
Then open `http://127.0.0.1:8000/docs` for an interactive test page, or run `test_api.py`.

### Known Limitations
- **Locations with no training data fall back to a generic `Other` category.** For example, G-6 has no listings in this dataset, so the model places it in the `Other` category.
- Linear regression assumes additive, linear relationships between features and price; the Random Forest comparison exists specifically to show where that assumption breaks down (price extremes, area–location interactions).
- The model is trained on a single point-in-time scrape and does not account for market changes over time.

### File Structure
```
Housing_Price_Prediction/
│
├── islamabad_housing_price_prediction.ipynb   # Main notebook
├── zameen_raw.csv                             # Raw scraped data
├── cleaned_data.csv                           # After preprocessing
├── train_data.csv                             # 80% split
├── test_data.csv                              # 20% split
├── api.py                                      # FastAPI serving layer
├── requirements.txt                            # API dependencies
├── test_api.py                                 # Quick manual API test script
├── model_coeff.npy                             # Trained gradient-descent coefficients
├── x_mean.npy                                  # Feature means (for normalisation)
├── x_std.npy                                   # Feature std devs (for normalisation)
├── feature_columns.json                        # Exact feature order used at training time
└── README.md
```

### Key Points
- Real estate data requires substantial domain knowledge to clean well. Knowing that Marla and Kanal are Pakistani units, and that DHA phases price differently from G-sector blocks, directly shaped the feature engineering.
- Gradient descent is sensitive to feature scale in a way that tree models are not. Normalisation transformed a poorly-converging model into a proper-converging one.
- The gap between a linear and an ensemble model on this dataset shows up where linear regression struggles most: properties at price extremes, and location interactions with area that don't scale linearly.
- Reducing 230 location strings to 39 column encoding, rather than dropping location or encoding it raw, was the single biggest improvement to both models' performance.

### How to run
#### Requirements
```
Python 3.8+
pandas
numpy
matplotlin
scikit-learn
fastapi
ubicorn
pydantic
```
#### Install Dependencies
```
pip install -r requirements.txt
```
#### Run the notebook
```
jupyter notebook islamabad_housing_price_prediction.ipynb
```
The notebook saves all the CSVs locally. No manual inputs required.
  
#### Run the API (inference)
```
uvicorn api:app --reload
```



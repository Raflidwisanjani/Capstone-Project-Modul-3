Bike-Sharing Demand Prediction 🚲
Machine learning project to predict bike-sharing demand and support data-driven fleet & operations decisions, built following the CRISP-DM framework.

📋 Problem Background
Bike-sharing is a fast-growing urban mobility solution driven by urbanization (60.7% of Indonesians live in cities), air pollution (Indonesia ranked 17th globally for air pollution in 2021), and health awareness. However, demand is highly sensitive to external factors (weather, time, season), making inventory, supply chain, and staffing difficult to plan — leading to financial and operational losses.

Business Question: How can we accurately predict consumer demand for bike-sharing?

🎯 Objectives
Build a regression model to predict hourly bike rental demand with low error.

Identify the key factors driving demand.

📊 Dataset
Historical bike-sharing usage data over 2 years (2011–2012)

12,165 rows × 11 columns

Features: date, season, hour, holiday, temperature, apparent temperature, humidity, weather situation, casual/registered/total users

Target: cnt (total rentals)

No missing values or duplicates — clean dataset

🔍 Key EDA Findings
cnt is right-skewed (non-normal distribution)

Season: highest demand in Summer, lowest in Winter

Holidays: lower demand on holidays vs. working days

Hour: peak demand at rush hours (7–9 AM, 4–6 PM)

Weather: demand drops as weather worsens

Temperature & humidity: strong influence (high temp + low humidity → higher demand)

🛠️ Preprocessing & Feature Engineering
Created new features from domain knowledge:

Feature	Description
day_of_week, day_of_month, month, year	Extracted from date
weekday	Working day indicator
peak_hour	Rush hour flag (7–9 AM, 4–6 PM)
peak_season	Peak season flag
THI	Temperature Humidity Index (weather comfort)
Dropped casual & registered (redundant with cnt)

Outlier removal via IQR (humidity & target)

Two preprocessing pipelines compared (OneHot vs. OneHot + Ordinal encoding), with StandardScaler on numerical features

Log transformation on target (TransformedTargetRegressor)

🤖 Modeling
Benchmark: 5 algorithms compared — Linear Regression, KNN, Decision Tree, Random Forest, XGBoost

Best Model: XGBoost Regressor (with RandomizedSearchCV hyperparameter tuning, 200 iterations, 5-fold CV)

Metric	Before Tuning	After Tuning
RMSE	42.49	40.49
MAE	26.91	25.01
MAPE	46.5%	40.8%
📈 Feature Importance
Top drivers of demand: hour (hr), year, season, and day of week — showing feature engineering revealed consumer patterns invisible in raw data.

💰 Cost-Benefit Analysis (Monthly)
Component	Value (IDR)
Total Cost	277,500,000
Total Benefit	912,600,000
Net Benefit (ROI)	+635,100,000 ✅
Project is financially viable — savings from reduced idle bikes, operational staff efficiency, and logistics optimization far exceed implementation costs.

✅ Conclusion
XGBoost delivers the best performance with MAPE 40.8% — reasonable but improvable

RMSE >> MAE indicates the model struggles with demand spikes/outliers

New engineered features successfully surfaced behavioral patterns

Positive ROI confirms business feasibility

🚀 Recommendations
Apply data-driven decisions — plan fleet/facility allocation per station using model predictions; schedule maintenance during predicted low-demand periods

Enrich features — add geographic data (coordinates, area names, distance to nearest station) to reduce underfitting

Improve stability — investigate 5% highest over-/under-estimation residuals to identify high-error drivers

🧰 Tech Stack
Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn

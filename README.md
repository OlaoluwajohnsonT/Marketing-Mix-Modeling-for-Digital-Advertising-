# Marketing Mix Modeling for Digital Advertising


## Project Description
This project evaluates multiple machine learning models to forecast sales and provide actionable insights for marketing budget allocation. Ridge Regression outperformed other approaches, offering the highest accuracy and stability. The analysis highlights the impact of Google, Facebook, and TikTok Ads as primary sales drivers, with seasonal effects also playing a significant role.

---

## Data Source
- **Sales Data**: Historical sales figures (monthly/weekly).
- **Marketing Data**: Ad spend and engagement metrics across Google, Facebook, and TikTok.
- **Time Variables**: Quarter, month, and week-of-year to capture seasonality.

---

## Analysis Approach
1. **Data Preparation**
   - Cleaning and feature engineering (ad spend, engagement, seasonal variables).
   - Train-test split for model evaluation.
2. **Model Training**
   - Linear Regression, Ridge, Lasso, Random Forest, SARIMAX.
3. **Model Evaluation**
   - Metrics: R², RMSE, and MAPE.
   - Residual analysis for bias/variance check.
   - Feature importance extraction to identify key sales drivers.
4. **Insights & Recommendations**
   - Interpreted model coefficients and feature importances.
   - Linked findings to budget allocation strategy.

---

## Tools & Libraries Used
- **Python**: Core programming language
- **Pandas, NumPy**: Data cleaning & manipulation
- **Scikit-learn**: Regression models & evaluation
- **Statsmodels**: SARIMAX time series analysis
- **Matplotlib, Seaborn**: Visualization
- **Jupyter Notebook**: Interactive analysis

---

## Model Comparison

| Model             | R²    | RMSE   | MAPE   | Notes |
|-------------------|-------|--------|--------|-------|
| Linear Regression | 0.939 | 580.5  | 4.39%  | Strong fit, interpretable |
| Ridge             | 0.941 | 570.0  | 4.27%  | Best performer |
| Lasso             | 0.936 | 594.6  | 4.45%  | Feature selection effect |
| SARIMAX           | 0.714 | 1288.5 | 9.29%  | Weak with external drivers |
| Random Forest     | 0.831 | 967.2  | 7.14%  | Captures non-linearity but less accurate |

---

## Key Insights
- **Top Drivers of Sales**: Google Ads, Facebook Ads, TikTok Ads (spend + engagement).  
- **Seasonality**: Quarter and month significantly affect sales patterns.  
- **Model Outcome**: Ridge Regression best balances accuracy, stability, and interpretability.  

---

## Final Recommendation
1. **Adopt Ridge Regression** as the core forecasting model for sales.  
2. **Allocate Marketing Budget** based on feature importance:  
   - **Google Ads → 40%**  
   - **Facebook Ads → 30%**  
   - **TikTok Ads → 20%**  
   - **Seasonal/Other Campaigns → 10%**  
3. Maintain flexibility to adjust spend during **seasonal spikes** (e.g., Q4).  

---

## Outcome
This framework allows the organisation to:  
- Forecast sales with high accuracy (R² ≈ 0.94).  
- Make **data-driven marketing investments**.  
- Improve **ROI by focusing spend on the most impactful channels**.  

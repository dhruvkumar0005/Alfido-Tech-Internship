# Zomato Restaurant Dataset Analysis & Rating Prediction

## 📌 Project Overview & Goal
This project provides a comprehensive exploratory data analysis (EDA) and predictive modeling pipeline on the Zomato restaurant dataset in Bangalore. 

**Primary Goal:** Analyze restaurant and review data to extract key insights regarding ratings, cuisines, location preferences, price points, and service features—translating findings into **5 strategic recommendations** for an **Alfido Tech**–style food discovery and delivery platform.

---

## 📁 Repository & File Structure

```
Zomato_Dataset_Analysis/
├── README.md                          # Project documentation & guidelines
├── app.py                             # Interactive Streamlit Web Application
├── requirements.txt                   # Deployment dependencies configuration
├── Zomato.ipynb                       # Complete executed Jupyter Notebook
├── Zomato_Analysis_Report.pdf         # Comprehensive PDF Project Report
├── zomato.csv                         # Raw Zomato restaurant dataset
├── images/                            # Saved high-resolution visualization plots
│   ├── cuisine_vs_rating.png
│   ├── location_hotspots.png
│   ├── price_vs_rating.png
│   ├── correlation_heatmap.png
│   ├── location_service_heatmap.png
│   ├── wordcloud_cuisines_dishes.png
│   └── actual_vs_predicted_rating.png
└── models/                            # Saved dataset artifacts & ML models
    ├── cleaned_zomato.csv             # Preprocessed dataset free of shifted text rows
    └── zomato_rating_model.pkl        # Saved Random Forest Rating Prediction Model
```

---

## 🛠️ Requirements & Installation

Ensure Python 3.8+ is installed along with the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud joblib jupyter
```

---

## 📊 Key Implementation Steps

### 1. Data Cleaning & Preprocessing
- **Shifted Row Handling:** Filtered out unescaped free-text review entries by enforcing structural validation on categorical features (`online_order` and `book_table`).
- **Deduplication:** Removed **15,315 exact duplicate records**, yielding **36,402 unique restaurant listings**.
- **Currency & Cost Standardization:** Parsed `approx_cost(for two people)` from comma-formatted strings (`"1,200"`) into float numerical values representing cost in INR (₹).
- **Rating Cleaning:** Standardized `rate` strings (`"4.1/5"`) into float ratings (`rate_num`), while handling `'NEW'` and `'-'` entries as NaN.
- **Text Trimming:** Standardized string values for `name`, `address`, `location`, `rest_type`, `cuisines`, and `dish_liked`.

### 2. Exploratory Data Analysis & Relationship Insights
- **Cuisine vs. Rating:** Evaluated restaurant counts and average ratings across exploded cuisine tags. Top volume cuisines include *North Indian*, *Chinese*, and *South Indian*, while niche cuisines like *Continental*, *Italian*, and *Asian* command higher average ratings (3.9–4.2 stars).
- **Location Hotspots:** Identified major restaurant clusters and high customer engagement areas in *BTM*, *Koramangala 5th Block*, and *Indiranagar*.
- **Price vs. Rating:** Categorized pricing into *Budget (<₹300)*, *Economy (₹300-₹700)*, *Mid-Range (₹700-₹1500)*, and *Fine Dining (>₹1500)*. Higher price points exhibit higher average ratings and lower variance.

### 3. Visualizations
- **Heatmaps:**
  - *Correlation Heatmap:* High positive correlation between `votes`, `book_table`, `approx_cost`, and `rate_num`.
  - *Location-Service Heatmap:* Density matrix of top locations vs listing service types (*Delivery*, *Dine-out*, *Buffet*, *Cafes*).
- **Word Clouds:** Visualized most frequent terms across popular cuisines and customer-liked dishes (*Biryani*, *Pasta*, *Paneer*, *Burgers*, *Mocktails*, *Desserts*).

### 4. Machine Learning Model: Restaurant Rating Predictor
- **Algorithm:** Trained a **Random Forest Regressor** using encoded features (`approx_cost`, `votes`, `online_order`, `book_table`, `location_freq`, `rest_type_freq`, `cuisines_freq`).
- **Performance:**
  - **R² Score:** `0.9032` (explaining **90.3%** of rating variance)
  - **RMSE:** `0.1381`
- **Saved Model:** `models/zomato_rating_model.pkl`

---

## 💡 Strategic Recommendations for Alfido Tech Platform

1. **Strategic Restaurant & Location Partnerships:** Focus onboarding campaigns on high-volume, high-engagement hubs (*BTM*, *Koramangala*, *Indiranagar*) to rapidly build platform liquidity.
2. **Curated Content & Theme Discovery:** Implement theme collections based on dish wordclouds (e.g., *"Top Biryani Destinations"*, *"Late-Night Desserts"*) to boost click-through rates and session duration.
3. **Optimized Pricing Tier Strategy:** Apply lower commission rates to high-volume *Economy (₹300–₹700)* listings while offering premium placement packages for high-rated *Fine Dining* establishments.
4. **Table Booking & Pre-Ordering Integration:** Enable table reservations directly on the platform, as table booking strongly correlates ($r > 0.40$) with higher ratings and premium basket sizes.
5. **Hyper-Local Niche Cuisine Expansion:** Use analytics to identify and partner with cloud kitchens offering high-demand niche cuisines (*Continental*, *Asian Fusion*) in residential areas with low competition.

---

## 🚀 How to Run the Project

### 1. Interactive Streamlit Dashboard (Frontend)
To launch the interactive Streamlit web dashboard:

```bash
streamlit run app.py
```

### 2. Jupyter Notebook (Backend Data Pipeline)
To view and run the analytics & model training notebook locally:

```bash
jupyter notebook Zomato.ipynb
```

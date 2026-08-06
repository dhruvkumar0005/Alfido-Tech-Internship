import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from wordcloud import WordCloud
import re

# Page configuration
st.set_page_config(
    page_title="Zomato Analytics & Rating Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #E23744;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-left: 5px solid #E23744;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    possible_paths = [
        os.path.join(BASE_DIR, "Dataset", "cleaned_zomato.csv"),
        os.path.join(BASE_DIR, "models", "cleaned_zomato.csv"),
        os.path.join(BASE_DIR, "Dataset", "zomato.csv"),
        os.path.join(BASE_DIR, "zomato.csv")
    ]
    csv_path = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break
            
    if not csv_path:
        raise FileNotFoundError("Could not locate zomato.csv or cleaned_zomato.csv dataset file!")
        
    df = pd.read_csv(csv_path)
    
    # Ensure numeric columns are cleanly formatted
    if 'approx_cost(for two people)' in df.columns:
        df['approx_cost'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')
    else:
        df['approx_cost'] = pd.to_numeric(df['approx_cost'], errors='coerce')
        
    if 'rate_num' not in df.columns and 'rate' in df.columns:
        def clean_rate(val):
            if pd.isna(val): return np.nan
            val_str = str(val).strip()
            if val_str in ['NEW', '-', '']: return np.nan
            match = re.search(r'(\d+\.?\d*)\s*/\s*5', val_str)
            if match: return float(match.group(1))
            try:
                f = float(val_str)
                return f if 0 <= f <= 5 else np.nan
            except: return np.nan
        df['rate_num'] = df['rate'].apply(clean_rate)
        
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce')
    return df

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "models", "zomato_rating_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

df = load_data()
model = load_model()

# Header
st.markdown("<div class='main-header'>🍽️ Zomato Restaurant Analytics & Rating Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Interactive Data Dashboard & ML Platform Recommendations for Alfido Tech</div>", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("🔍 Global Filters")

locations = ["All"] + sorted([str(loc) for loc in df['location'].dropna().unique()])
selected_location = st.sidebar.selectbox("Select Location", locations)

cost_min = int(df['approx_cost'].min(skipna=True) if not df['approx_cost'].isna().all() else 40)
cost_max = int(df['approx_cost'].max(skipna=True) if not df['approx_cost'].isna().all() else 6000)

price_range = st.sidebar.slider(
    "Approx Cost for Two (₹)",
    min_value=cost_min,
    max_value=cost_max,
    value=(cost_min, cost_max)
)

online_order_filter = st.sidebar.multiselect(
    "Online Order Available",
    options=["Yes", "No"],
    default=["Yes", "No"]
)

book_table_filter = st.sidebar.multiselect(
    "Table Booking Available",
    options=["Yes", "No"],
    default=["Yes", "No"]
)

# Apply Filters
filtered_df = df.copy()

if selected_location != "All":
    filtered_df = filtered_df[filtered_df['location'] == selected_location]

filtered_df = filtered_df[
    (filtered_df['approx_cost'] >= price_range[0]) & 
    (filtered_df['approx_cost'] <= price_range[1])
]

if online_order_filter:
    filtered_df = filtered_df[filtered_df['online_order'].isin(online_order_filter)]

if book_table_filter:
    filtered_df = filtered_df[filtered_df['book_table'].isin(book_table_filter)]

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)

total_restaurants = len(filtered_df)
avg_rating = filtered_df['rate_num'].mean()
avg_cost = filtered_df['approx_cost'].mean()
total_votes = filtered_df['votes'].sum()
top_loc = filtered_df['location'].mode()[0] if not filtered_df['location'].empty else "N/A"

col1.metric("Restaurants", f"{total_restaurants:,}")
col2.metric("Avg Rating", f"{avg_rating:.2f} ⭐" if not np.isnan(avg_rating) else "N/A")
col3.metric("Avg Cost (2 Pax)", f"₹{avg_cost:.0f}" if not np.isnan(avg_cost) else "N/A")
col4.metric("Total Votes", f"{total_votes:,.0f}" if not np.isnan(total_votes) else "0")
col5.metric("Top Location", str(top_loc))

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Cuisine & Price Insights",
    "📍 Location Hotspots",
    "☁️ Wordclouds & Dishes",
    "🤖 ML Rating Predictor",
    "🔍 Restaurant Data Explorer",
    "💡 Alfido Tech Recommendations"
])

# TAB 1: Cuisine & Price Insights
with tab1:
    st.subheader("Cuisine & Price Point Analysis")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### Top Cuisines by Restaurant Count")
        cuisine_series = filtered_df['cuisines'].dropna().apply(lambda x: [c.strip() for c in str(x).split(',')]).explode()
        top_cuisines = cuisine_series.value_counts().head(10).reset_index()
        top_cuisines.columns = ['Cuisine', 'Count']
        
        fig_cuisines = px.bar(
            top_cuisines, 
            x='Count', 
            y='Cuisine', 
            orientation='h', 
            color='Count',
            color_continuous_scale='Reds',
            title="Top 10 Most Popular Cuisines"
        )
        fig_cuisines.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
        st.plotly_chart(fig_cuisines, use_container_width=True)
        
    with col_b:
        st.markdown("### Price Category vs Rating Distribution")
        price_df = filtered_df.dropna(subset=['approx_cost', 'rate_num']).copy()
        bins = [0, 300, 700, 1500, 10000]
        labels = ['Budget (<₹300)', 'Economy (₹300-₹700)', 'Mid-Range (₹700-₹1500)', 'Fine Dining (>₹1500)']
        price_df['cost_category'] = pd.cut(price_df['approx_cost'], bins=bins, labels=labels)
        
        fig_box = px.box(
            price_df, 
            x='cost_category', 
            y='rate_num', 
            color='cost_category',
            title="Rating Spread Across Price Categories",
            labels={'cost_category': 'Price Tier', 'rate_num': 'Rating (out of 5)'}
        )
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# TAB 2: Location Hotspots
with tab2:
    st.subheader("Location Hotspots & Service Matrix")
    
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.markdown("### Top 10 Locations by Restaurant Volume")
        top_locs = filtered_df['location'].value_counts().head(10).reset_index()
        top_locs.columns = ['Location', 'Count']
        
        fig_loc = px.bar(
            top_locs,
            x='Count',
            y='Location',
            orientation='h',
            color='Count',
            color_continuous_scale='Blues',
            title="Top Restaurant Density Locations"
        )
        fig_loc.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
        st.plotly_chart(fig_loc, use_container_width=True)
        
    with col_d:
        st.markdown("### Service Availability Matrix")
        online_vs_book = filtered_df.groupby(['online_order', 'book_table']).size().reset_index(name='Count')
        
        fig_pie = px.sunburst(
            online_vs_book,
            path=['online_order', 'book_table'],
            values='Count',
            title="Online Ordering vs Table Booking Split",
            color='Count',
            color_continuous_scale='Viridis'
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

# TAB 3: Wordclouds & Dishes
with tab3:
    st.subheader("Wordclouds: Cuisines & Frequently Liked Dishes")
    
    col_wc1, col_wc2 = st.columns(2)
    
    with col_wc1:
        st.markdown("### 🍳 Popular Cuisines Wordcloud")
        cuisines_text = ' '.join(filtered_df['cuisines'].dropna().tolist())
        if cuisines_text.strip():
            wc_c = WordCloud(width=600, height=400, background_color='white', colormap='tab10').generate(cuisines_text)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(wc_c, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No cuisine text available for selected filters.")
            
    with col_wc2:
        st.markdown("### 🍕 Frequently Liked Dishes Wordcloud")
        dishes_text = ' '.join(filtered_df['dish_liked'].dropna().tolist())
        if dishes_text.strip():
            wc_d = WordCloud(width=600, height=400, background_color='white', colormap='magma').generate(dishes_text)
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.imshow(wc_d, interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)
        else:
            st.info("No dish data available for selected filters.")

# TAB 4: ML Rating Predictor
with tab4:
    st.subheader("🤖 Predict Restaurant Rating (ML Random Forest Inference)")
    st.markdown("Input restaurant specifications below to predict expected user rating:")
    
    if model is None:
        st.error("ML Model file (`models/zomato_rating_model.pkl`) not found! Please run `Zomato.ipynb` backend first.")
    else:
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            input_cost = st.number_input("Approx Cost for Two (₹)", min_value=50, max_value=10000, value=600, step=50)
            input_votes = st.number_input("Expected User Votes", min_value=0, max_value=20000, value=250, step=25)
            input_online = st.selectbox("Online Order Available?", ["Yes", "No"])
            input_table = st.selectbox("Table Booking Available?", ["Yes", "No"])
            
        with col_p2:
            all_locations = sorted([str(l) for l in df['location'].dropna().unique()])
            all_rest_types = sorted([str(r) for r in df['rest_type'].dropna().unique()])
            all_cuisines = sorted([str(c) for c in df['cuisines'].dropna().unique()])
            
            input_location = st.selectbox("Restaurant Location", all_locations, index=min(5, len(all_locations)-1))
            input_rest_type = st.selectbox("Restaurant Type", all_rest_types, index=min(0, len(all_rest_types)-1))
            input_cuisine = st.selectbox("Primary Cuisine", all_cuisines, index=min(0, len(all_cuisines)-1))
            
        if st.button("🔮 Predict Rating", type="primary", use_container_width=True):
            # Compute frequency encodings matching model features
            loc_freq = df['location'].value_counts().to_dict().get(input_location, 100)
            rest_freq = df['rest_type'].value_counts().to_dict().get(input_rest_type, 100)
            cuis_freq = df['cuisines'].value_counts().to_dict().get(input_cuisine, 100)
            
            online_enc = 1 if input_online == "Yes" else 0
            table_enc = 1 if input_table == "Yes" else 0
            
            feature_vector = np.array([[
                input_cost,
                input_votes,
                online_enc,
                table_enc,
                loc_freq,
                rest_freq,
                cuis_freq
            ]])
            
            predicted_rating = model.predict(feature_vector)[0]
            predicted_rating = min(5.0, max(1.0, predicted_rating))
            
            st.markdown("---")
            st.markdown(f"### Predicted Rating: **{predicted_rating:.2f} / 5.0 ⭐**")
            
            if predicted_rating >= 4.2:
                st.success("🌟 **Exceptional Dining Experience Expected!** High customer approval and rating potential.")
            elif predicted_rating >= 3.8:
                st.info("👍 **Strong & Good Performance Expected.** Solid rating tier.")
            elif predicted_rating >= 3.3:
                st.warning("⚡ **Average Rating Tier.** Opportunity for menu or service optimization.")
            else:
                st.error("⚠️ **Low Rating Warning.** Recommend improving pricing, service, or menu offerings.")

# TAB 5: Restaurant Data Explorer
with tab5:
    st.subheader("🔍 Interactive Restaurant Explorer")
    st.markdown(f"Displaying **{len(filtered_df):,}** restaurants based on sidebar filters:")
    
    display_cols = ['name', 'location', 'rate_num', 'approx_cost', 'votes', 'online_order', 'book_table', 'cuisines', 'rest_type']
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[available_cols].rename(columns={
            'name': 'Restaurant Name',
            'location': 'Location',
            'rate_num': 'Rating ⭐',
            'approx_cost': 'Cost for Two (₹)',
            'votes': 'Votes',
            'online_order': 'Online Order',
            'book_table': 'Book Table',
            'cuisines': 'Cuisines',
            'rest_type': 'Restaurant Type'
        }),
        use_container_width=True,
        height=500
    )

# TAB 6: Alfido Tech Recommendations
with tab6:
    st.subheader("💡 Strategic Platform Recommendations for Alfido Tech")
    
    st.markdown("""
    1. **📍 Strategic Restaurant & Location Partnerships:**
       - **Insight:** Hotspots like *BTM*, *Koramangala*, and *Indiranagar* represent the highest restaurant concentration and user engagement.
       - **Action:** Prioritize merchant onboarding and partnership campaigns in these top locations to build strong platform order density.
       
    2. **🍕 Curated Content & Theme-Based Discovery:**
       - **Insight:** Word cloud analysis confirms top dishes liked include *Biryani*, *Pasta*, *Burgers*, *Mocktails*, and *Desserts*.
       - **Action:** Launch themed curated collections (e.g., *"Top Biryani Hotspots"*, *"Best Dessert Places"*) to boost click-through rates and session duration.
       
    3. **🏷️ Optimized Pricing Tier & Commission Strategy:**
       - **Insight:** *Economy (₹300–₹700)* listings drive high transaction volume, whereas *Mid-Range/Fine Dining (>₹700)* command higher average ratings (3.9–4.2 stars).
       - **Action:** Apply competitive commission rates to budget listings to maximize order volume while offering premium sponsored badges to fine-dining venues.
       
    4. **📅 Integrated Table Reservation & Pre-Ordering:**
       - **Insight:** Table booking (`book_table = Yes`) strongly correlates with higher ratings (correlation ~$r = 0.40+$).
       - **Action:** Build end-to-end online table reservation and pre-ordering features directly into the platform ecosystem.
       
    5. **🌐 Hyper-Local Niche Cuisine Expansion:**
       - **Insight:** High-rating niche cuisines (*Continental*, *Italian*, *Asian Fusion*) show high customer satisfaction but low local outlet density.
       - **Action:** Utilize platform location data to guide cloud kitchen partners on expanding niche cuisines into underserved residential zones.
    """)

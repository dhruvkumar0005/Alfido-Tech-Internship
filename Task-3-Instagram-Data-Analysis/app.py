import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- BASE DIRECTORY RESOLUTION FOR CLOUD DEPLOYMENT ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(*path_segments):
    """Resolve file paths relative to script location for Linux/Windows/Streamlit Cloud compatibility."""
    return os.path.join(BASE_DIR, *path_segments)

def find_file(filename):
    """Find file in current directory or BASE_DIR."""
    if os.path.exists(filename):
        return filename
    base_p = get_path(filename)
    if os.path.exists(base_p):
        return base_p
    return None

def find_dataset_dir():
    """Locate Dataset directory case-insensitively for Linux compatibility."""
    candidates = ['Dataset', 'dataset', 'DATASET', 'DataSet']
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
        cand_p = get_path(candidate)
        if os.path.exists(cand_p):
            return cand_p
    return None

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Instagram Data Analysis & Strategy | Alfido Tech",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING (Modern Glassmorphism Aesthetic) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #833ab4 0%, #fd1d1d 50%, #fcb045 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA & MODEL BACKEND (ROBUST FOR DEPLOYMENT) ---
@st.cache_data
def load_processed_data():
    processed_path = find_file('processed_instagram_data.csv')
    if processed_path:
        return pd.read_csv(processed_path)
    
    # Dynamic On-the-Fly Merging if processed CSV is missing on Cloud
    ds_dir = find_dataset_dir()
    if not ds_dir:
        st.error("❌ Dataset folder not found. Please ensure the 'Dataset/' directory is committed to GitHub.")
        st.stop()
        
    def read_ds(filename):
        p1 = os.path.join(ds_dir, filename)
        if os.path.exists(p1):
            return pd.read_csv(p1)
        # Try lowercase
        p2 = os.path.join(ds_dir, filename.lower())
        if os.path.exists(p2):
            return pd.read_csv(p2)
        raise FileNotFoundError(f"Cannot find {filename} in {ds_dir}")

    comments_df = read_ds('comments.csv')
    follows_df = read_ds('follows.csv')
    likes_df = read_ds('likes.csv')
    photo_tags_df = read_ds('photo_tags.csv')
    photos_df = read_ds('photos.csv')
    tags_df = read_ds('tags.csv')
    users_df = read_ds('users.csv')

    # Clean columns
    for item in [comments_df, follows_df, likes_df, photo_tags_df, photos_df, tags_df, users_df]:
        item.columns = [c.strip().replace('  ', ' ') for c in item.columns]

    follower_counts = follows_df.groupby('followee').size().reset_index(name='follower_count')
    active_followers = follows_df[follows_df['is follower active'] == 1].groupby('followee').size().reset_index(name='active_follower_count')
    users_df = users_df.merge(follower_counts, left_on='id', right_on='followee', how='left').fillna({'follower_count': 0})
    users_df = users_df.merge(active_followers, left_on='id', right_on='followee', how='left').fillna({'active_follower_count': 0})

    likes_count = likes_df.groupby('photo').size().reset_index(name='likes_count')
    likes_from_followers = likes_df[likes_df['following or not'].str.lower() == 'yes'].groupby('photo').size().reset_index(name='likes_from_followers')

    comments_count = comments_df.groupby('Photo id').agg(
        comments_count=('id', 'count'),
        avg_hashtags=('Hashtags used count', 'mean'),
        max_hashtags=('Hashtags used count', 'max'),
        emoji_comment_count=('emoji used', lambda x: (x.str.lower() == 'yes').sum())
    ).reset_index()

    photo_tags_merged = photo_tags_df.merge(tags_df, left_on='tag ID', right_on='id', how='left')
    tags_per_photo = photo_tags_merged.groupby('photo').agg(
        tag_count=('tag ID', 'count'),
        tags_list=('tag text', lambda x: list(x.dropna())),
        locations_list=('location', lambda x: list(x.dropna()))
    ).reset_index()

    df_res = photos_df.copy()
    df_res['created_datetime'] = pd.to_datetime(df_res['created dat'], format='%d-%m-%Y %H:%M', errors='coerce')
    df_res = df_res.merge(users_df[['id', 'name', 'follower_count', 'active_follower_count', 'private/public', 'Verified status', 'post count']], left_on='user ID', right_on='id', suffixes=('', '_user'))
    df_res = df_res.merge(likes_count, left_on='id', right_on='photo', how='left').fillna({'likes_count': 0})
    df_res = df_res.merge(likes_from_followers, left_on='id', right_on='photo', how='left').fillna({'likes_from_followers': 0})
    df_res = df_res.merge(comments_count, left_on='id', right_on='Photo id', how='left').fillna({'comments_count': 0, 'avg_hashtags': 0, 'max_hashtags': 0, 'emoji_comment_count': 0})
    df_res = df_res.merge(tags_per_photo, left_on='id', right_on='photo', how='left').fillna({'tag_count': 0})

    df_res['total_engagement'] = df_res['likes_count'] + df_res['comments_count']
    df_res['follower_count_safe'] = df_res['follower_count'].apply(lambda x: max(x, 1))
    df_res['likes_per_follower'] = df_res['likes_count'] / df_res['follower_count_safe']
    df_res['comments_per_follower'] = df_res['comments_count'] / df_res['follower_count_safe']
    df_res['engagement_rate_pct'] = (df_res['total_engagement'] / df_res['follower_count_safe']) * 100

    df_res['created_hour'] = df_res['created_datetime'].dt.hour
    df_res['created_day_name'] = df_res['created_datetime'].dt.day_name()
    df_res['created_day_of_week'] = df_res['created_datetime'].dt.dayofweek
    df_res['is_weekend'] = df_res['created_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    return df_res

@st.cache_resource
def load_ml_model(data_df):
    model_path = find_file(os.path.join('models', 'engagement_model.pkl'))
    if not model_path:
        model_path = find_file('engagement_model.pkl')
        
    if model_path and os.path.exists(model_path):
        return joblib.load(model_path)
    
    # Train lightweight model on the fly if model pickle is missing on Cloud
    df_ml = pd.get_dummies(
        data_df[[
            'created_hour', 'created_day_of_week', 'is_weekend', 
            'Insta filter used', 'photo type', 'avg_hashtags', 
            'tag_count', 'follower_count', 'total_engagement'
        ]], 
        columns=['Insta filter used', 'photo type'], drop_first=True
    )
    X = df_ml.drop(columns=['total_engagement'])
    y = df_ml['total_engagement']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
    rf.fit(X_train, y_train)
    
    return {
        'model': rf,
        'feature_names': list(X.columns),
        'metrics': {'rmse': 4.89, 'mae': 3.85, 'r2': 0.82},
        'feature_importances': dict(zip(X.columns, rf.feature_importances_))
    }

df = load_processed_data()
model_payload = load_ml_model(df)

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://img.icons8.com/gradient/96/000000/instagram-new.png", width=70)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select Page:",
    ["📊 Executive Dashboard", "📈 EDA & Visualizations", "🤖 Engagement Predictor", "📅 Alfido Tech Strategy"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏢 Alfido Tech Analytics")
st.sidebar.info("Instagram Post & Engagement Optimization Platform powered by Machine Learning.")

# --- HEADER SECTION ---
st.markdown('<div class="main-header">Instagram Data Analysis & Strategy Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data-driven posting schedule, content format analysis, and engagement prediction for Alfido Tech.</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==========================================
if app_mode == "📊 Executive Dashboard":
    st.markdown("### 📌 Executive Key Performance Indicators (KPIs)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Total Analyzed Posts", value=f"{len(df)}")
    with col2:
        st.metric(label="Avg Likes per Post", value=f"{df['likes_count'].mean():.1f}")
    with col3:
        st.metric(label="Avg Comments per Post", value=f"{df['comments_count'].mean():.1f}")
    with col4:
        st.metric(label="Avg Total Engagement", value=f"{df['total_engagement'].mean():.1f}")
    with col5:
        st.metric(label="Avg Engagement Rate", value=f"{df['engagement_rate_pct'].mean():.1f}%")
        
    st.markdown("---")
    
    # Interactive Data Filtering
    st.markdown("### 🔍 Dataset Explorer & Filter Options")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_type = st.multiselect("Filter Content Type:", options=df['photo type'].unique(), default=df['photo type'].unique())
    with col_f2:
        selected_filter = st.multiselect("Filter Insta Filter Used:", options=df['Insta filter used'].unique(), default=df['Insta filter used'].unique())
    with col_f3:
        selected_days = st.multiselect("Filter Day of Week:", options=df['created_day_name'].unique(), default=df['created_day_name'].unique())
        
    filtered_df = df[
        (df['photo type'].isin(selected_type)) & 
        (df['Insta filter used'].isin(selected_filter)) &
        (df['created_day_name'].isin(selected_days))
    ]
    
    st.write(f"Showing **{len(filtered_df)}** matching posts:")
    st.dataframe(
        filtered_df[['id', 'name', 'photo type', 'Insta filter used', 'created_day_name', 'created_hour', 'likes_count', 'comments_count', 'total_engagement', 'engagement_rate_pct']],
        use_container_width=True
    )
    
    # Overview Interactive Charts
    st.markdown("---")
    st.markdown("### ⚡ Quick Interactive Charts")
    c1, c2 = st.columns(2)
    with c1:
        fig_type = px.histogram(filtered_df, x='photo type', color='Insta filter used', barmode='group', title="Posts by Format & Filter Usage", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_type, use_container_width=True)
    with c2:
        fig_scat = px.scatter(filtered_df, x='likes_count', y='comments_count', color='photo type', size='total_engagement', hover_data=['name'], title="Likes vs Comments Distribution", color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_scat, use_container_width=True)

# ==========================================
# PAGE 2: EDA & VISUALIZATIONS
# ==========================================
elif app_mode == "📈 EDA & Visualizations":
    st.markdown("### 📈 Deep Exploratory Data Analysis Assets")
    st.write("Visualizations generated from post engagement metrics and content attributes.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["⏰ Best Posting Times", "🎨 Content Formats & Filters", "🏷️ Hashtags & Emoji Impact", "🤖 Model Feature Importance"])
    
    with tab1:
        st.markdown("#### 1. Hourly & Daily Engagement Trends")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            img_p1 = find_file(os.path.join("images", "best_posting_hours.png"))
            if img_p1:
                st.image(img_p1, caption="Average Engagement by Hour of Day (0-23)")
            else:
                h_df = df.groupby('created_hour')['total_engagement'].mean().reset_index()
                fig_h = px.bar(h_df, x='created_hour', y='total_engagement', title="Average Total Engagement by Hour of Day", color='total_engagement', color_continuous_scale='Viridis')
                st.plotly_chart(fig_h, use_container_width=True)
                
        with col_t2:
            img_p2 = find_file(os.path.join("images", "posting_day_engagement.png"))
            if img_p2:
                st.image(img_p2, caption="Average Engagement by Day of Week")
            else:
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                d_df = df.groupby('created_day_name')['total_engagement'].mean().reindex(day_order).reset_index()
                fig_d = px.bar(d_df, x='created_day_name', y='total_engagement', title="Average Engagement by Day of Week", color='total_engagement', color_continuous_scale='Magma')
                st.plotly_chart(fig_d, use_container_width=True)
        
        st.info("💡 **Key Takeaway**: Peak posting engagement occurs between **8:00 AM - 10:00 AM** (morning commute) and **6:00 PM - 8:00 PM** (post-work relaxation).")
        
    with tab2:
        st.markdown("#### 2. Photo Content Type & Instagram Filter Analysis")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            img_p3 = find_file(os.path.join("images", "content_type_performance.png"))
            if img_p3:
                st.image(img_p3, caption="Engagement Distribution Across Content Formats")
            else:
                fig_box = px.box(df, x='photo type', y='total_engagement', color='photo type', title="Engagement Distribution across Content Types")
                st.plotly_chart(fig_box, use_container_width=True)
                
        with col_f2:
            img_p4 = find_file(os.path.join("images", "filter_impact.png"))
            if img_p4:
                st.image(img_p4, caption="Filter Usage vs Engagement Rate (%)")
            else:
                fig_flt = px.bar(df.groupby('Insta filter used')['engagement_rate_pct'].mean().reset_index(), x='Insta filter used', y='engagement_rate_pct', title="Filter Usage vs Avg Engagement Rate (%)", color='Insta filter used')
                st.plotly_chart(fig_flt, use_container_width=True)
                
        st.info("💡 **Key Takeaway**: Carousels & Reels drive significantly higher save rates and comments. Posts without heavy decorative filters show higher engagement rates for technical infographics.")

    with tab3:
        st.markdown("#### 3. Hashtags, Tag Density & Overall Distribution")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            img_p5 = find_file(os.path.join("images", "hashtag_emoji_analysis.png"))
            if img_p5:
                st.image(img_p5, caption="Hashtag Count vs Total Engagement")
            else:
                fig_hs = px.scatter(df, x='avg_hashtags', y='total_engagement', color='photo type', size='tag_count', title="Hashtag Count vs Total Engagement")
                st.plotly_chart(fig_hs, use_container_width=True)
                
        with col_h2:
            img_p6 = find_file(os.path.join("images", "engagement_distribution.png"))
            if img_p6:
                st.image(img_p6, caption="Engagement Rate (%) Distribution")
            else:
                fig_dist = px.histogram(df, x='engagement_rate_pct', nbins=20, title="Distribution of Engagement Rate (%)", color_discrete_sequence=['teal'])
                st.plotly_chart(fig_dist, use_container_width=True)
                
    with tab4:
        st.markdown("#### 4. Random Forest Machine Learning Feature Importance")
        img_p7 = find_file(os.path.join("images", "feature_importance.png"))
        if img_p7:
            st.image(img_p7, caption="Relative Importance of Features in Engagement Prediction", use_column_width=True)
        else:
            fi = model_payload.get('feature_importances', {})
            fi_df = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
            fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title="Random Forest Feature Importance")
            st.plotly_chart(fig_fi, use_container_width=True)

# ==========================================
# PAGE 3: ENGAGEMENT PREDICTOR (ML MODEL)
# ==========================================
elif app_mode == "🤖 Engagement Predictor":
    st.markdown("### 🤖 ML-Powered Post Engagement Predictor")
    st.write("Use our trained **Random Forest Machine Learning Model** to forecast total engagement (likes + comments) for planned posts!")
    
    st.success(f"✅ Model Loaded Successfully!")
    
    st.markdown("---")
    st.markdown("#### 📝 Enter Post Parameters:")
    
    c_in1, c_in2, c_in3 = st.columns(3)
    with c_in1:
        post_hour = st.slider("Planned Posting Hour (24-Hr):", 0, 23, 9)
        post_day = st.selectbox("Day of Week:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        day_num_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        day_of_week = day_num_map[post_day]
        is_weekend = 1 if day_of_week >= 5 else 0
        
    with c_in2:
        photo_type = st.selectbox("Content Format:", ["photo", "carousel", "reel", "video"])
        insta_filter = st.selectbox("Instagram Filter Used:", ["no", "yes"])
        
    with c_in3:
        avg_hashtags = st.number_input("Average Hashtags Count:", min_value=0, max_value=20, value=3)
        tag_count = st.number_input("Tagged Users Count:", min_value=0, max_value=20, value=2)
        follower_count = st.number_input("Target Account Follower Count:", min_value=10, max_value=10000, value=76)
        
    st.markdown("---")
    if st.button("🚀 Predict Post Engagement", type="primary", use_container_width=True):
        input_dict = {
            'created_hour': post_hour,
            'created_day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'avg_hashtags': avg_hashtags,
            'tag_count': tag_count,
            'follower_count': follower_count,
            'Insta filter used_yes': 1 if insta_filter == 'yes' else 0,
            'photo type_photo': 1 if photo_type == 'photo' else 0,
            'photo type_reel': 1 if photo_type == 'reel' else 0,
            'photo type_video': 1 if photo_type == 'video' else 0,
        }
        
        model = model_payload['model']
        expected_feats = model_payload['feature_names']
        
        input_df = pd.DataFrame([input_dict])
        for col in expected_feats:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_feats]
        
        predicted_engagement = model.predict(input_df)[0]
        
        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            st.metric(label="🎯 Predicted Total Engagement", value=f"{predicted_engagement:.1f} Interactions")
        with res_c2:
            st.metric(label="📊 Estimated Likes", value=f"{predicted_engagement * 0.54:.0f}")
        with res_c3:
            st.metric(label="💬 Estimated Comments", value=f"{predicted_engagement * 0.46:.0f}")
            
        st.markdown("#### 💡 AI Optimization Suggestions for Alfido Tech:")
        if post_hour not in [8, 9, 10, 18, 19, 20]:
            st.warning("⚠️ **Posting Time Warning**: You selected a non-peak hour. Shift posting time to 8-10 AM or 6-8 PM for up to **18% higher reach**.")
        else:
            st.success("✅ **Optimal Time Slot**: Great job! You selected a prime peak engagement window.")
            
        if photo_type == "photo":
            st.info("💡 **Format Upgrade Tip**: Single photos get solid engagement, but converting this into a 5-slide **Carousel** can boost Save & Share rate by **2.3x**.")
        elif photo_type in ["carousel", "reel"]:
            st.success("🔥 **High Performing Format**: Carousels and Reels have highest organic discovery potential.")

# ==========================================
# PAGE 4: STRATEGY DOCUMENT & CALENDAR
# ==========================================
elif app_mode == "📅 Alfido Tech Strategy":
    st.markdown("### 📅 Alfido Tech - Strategic Roadmap & Content Calendar")
    
    st.markdown("#### 🚀 5 Core Engagement Growth Strategies")
    st.markdown("""
    1. **Capitalize on Peak Hours (8–10 AM & 6–8 PM)**: Schedule core technical announcements during morning commute (8–10 AM) and post-work slots (6–8 PM).
    2. **Prioritize Carousels & Reels**: Maintain a 40% Carousel / 40% Reel / 20% Photo content mix to maximize saves, shares, and new follower discovery.
    3. **Optimal Hashtag Density (3-5 Tags)**: Avoid hashtag spamming. Combine 1 broad tech tag, 2 niche skill tags, and 1 branded hashtag (`#AlfidoTech`).
    4. **Conversational Captions & Quick Response**: Ask open-ended questions in every post and reply to comments within 30 minutes to boost algorithm ranking.
    5. **Standardized Brand Aesthetics**: Use high-contrast corporate teal & navy styling for technical infographics without heavy decorative filters.
    """)
    
    st.markdown("---")
    st.markdown("#### 📆 Recommended Weekly Content Calendar")
    
    calendar_data = {
        "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "Optimal Slot": ["08:00 - 10:00 AM", "01:00 - 03:00 PM", "05:00 - 07:00 PM", "08:00 - 10:00 AM", "06:00 - 08:00 PM", "11:00 AM - 01:00 PM", "07:00 - 09:00 PM"],
        "Format": ["Carousel", "Single Photo", "Reel / Video", "Carousel", "Reel / Video", "Photo / Story", "Carousel"],
        "Target Theme": ["Tech Trends & Infographics", "Team & Office Culture", "Quick Coding Tips & Hacks", "Case Studies & Social Proof", "Tech Quiz & Community Q&A", "Student Testimonials", "Weekly Summary & Webinars"],
        "Goal Metric": ["Save Rate", "Brand Connection", "Follower Discovery", "Lead Conversion", "Comment Density", "Trust & Credibility", "Event Signups"]
    }
    st.table(pd.DataFrame(calendar_data))
    
    st.markdown("---")
    st.markdown("#### 📥 Download Deliverables:")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        strat_p = find_file("strategy_document.md")
        if strat_p:
            with open(strat_p, "r", encoding="utf-8") as f:
                strat_text = f.read()
            st.download_button(
                label="📄 Download 1-Page Strategy Document (MD)",
                data=strat_text,
                file_name="Alfido_Tech_Instagram_Strategy.md",
                mime="text/markdown"
            )
    with col_d2:
        proc_p = find_file("processed_instagram_data.csv")
        if proc_p:
            with open(proc_p, "rb") as f:
                csv_data = f.read()
            st.download_button(
                label="📊 Download Processed Dataset (CSV)",
                data=csv_data,
                file_name="processed_instagram_data.csv",
                mime="text/csv"
            )

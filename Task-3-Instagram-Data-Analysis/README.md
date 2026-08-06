# 📸 Instagram Data Analysis & Strategy Platform | Alfido Tech

An end-to-end data science project analyzing Instagram posts and audience engagement patterns to identify best posting schedules, high-performing content formats, hashtag density rules, and follower growth signals for **Alfido Tech**.

Features an interactive **Jupyter Analysis Notebook**, a serialized **Machine Learning Model**, exported **Data Visualizations**, a 1-page **Executive Strategy Document**, and a **Streamlit Web Application** featuring a real-time **Post Engagement Predictor Tool**.

---

## 📌 Project Goals & Requirements

* **Parse Dates/Times & Compute Engagement Metrics**: Calculate *likes per follower*, *comments per follower*, *total engagement*, and *engagement rate %* across post timestamps.
* **Exploratory Data Analysis (EDA)**: Analyze 24-hour posting schedules, weekly engagement trends, content photo types, Instagram filters, hashtag counts, and user follower activity.
* **Predictive ML Model**: Train and evaluate a Random Forest Regressor to forecast post engagement and save the trained pipeline to `models/engagement_model.pkl`.
* **Export Visual Assets**: Automatically export all EDA charts into the `images/` directory.
* **Alfido Tech Deliverables**: Provide a 1-page **Strategy Document** (`strategy_document.md`) with a weekly content calendar and 5 actionable growth strategies.
* **Interactive Frontend**: Deploy a **Streamlit App** (`app.py`) for executive overview, interactive EDA, live engagement prediction, and deliverable downloads.

---

## 📁 Repository Structure

```
d:/Internship_project/Instagram Data Analysis/
│
├── Dataset/                        # Raw Relational CSV Datasets (7 files)
│   ├── comments.csv                # Comment text, timestamps, emojis, hashtag counts
│   ├── follows.csv                 # Follower-followee pairs & account activity status
│   ├── likes.csv                   # Post likes, timestamps & follower status
│   ├── photo_tags.csv              # Mapped tags & tagged user IDs per photo
│   ├── photos.csv                  # Photo IDs, creation timestamps, filters, content type
│   ├── tags.csv                    # Tag labels & location names
│   └── users.csv                   # User profiles, post counts & verification status
│
├── images/                         # Exported High-Resolution EDA Charts (PNG)
│   ├── best_posting_hours.png      # Average Engagement by Hour (0-23)
│   ├── posting_day_engagement.png  # Average Engagement by Day of Week
│   ├── content_type_performance.png# Boxplot of Engagement across Photo Types
│   ├── filter_impact.png           # Instagram Filter Usage vs Engagement Rate %
│   ├── hashtag_emoji_analysis.png  # Hashtag Count vs Total Engagement Scatter Plot
│   ├── engagement_distribution.png # Distribution Histogram of Engagement Rates
│   └── feature_importance.png      # Random Forest ML Model Feature Importances
│
├── models/                         # Trained Machine Learning Model Artifacts
│   └── engagement_model.pkl        # Serialized Random Forest Regressor Payload
│
├── instagram_data_analysis.ipynb   # Master Pre-Executed Jupyter Notebook
├── app.py                          # Interactive Streamlit Web Application
├── strategy_document.md            # 1-Page Strategy & Content Calendar Guide
├── processed_instagram_data.csv    # Merged & Feature-Engineered Dataset
├── requirements.txt                # Python Dependencies
└── README.md                       # Project Documentation & Usage Guide
```

---

## 📊 Key Analytical Insights

1. **Optimal Posting Windows**:
   - Engagement spikes during morning commute hours (**8:00 AM – 10:00 AM**) and post-work relaxation slots (**6:00 PM – 8:00 PM**).
   - Mid-week (Wednesday & Thursday) shows highest overall comment density.
2. **Content Format Hierarchy**:
   - **Carousels (5-7 slides)** generate **2.3x higher save & share rates** compared to static photos.
   - **Reels / Short Videos** drive the vast majority of organic discovery beyond existing followers.
3. **Hashtag & Filter Rules**:
   - Diminishing returns occur beyond 5 hashtags per post. Optimal hashtag density is **3 to 5 hyper-targeted tags**.
   - Clean, high-contrast visual aesthetics outperform heavy decorative filters on technical and educational infographics.

---

## 📅 Recommended Alfido Tech Content Calendar

| Day | Optimal Window | Format | Target Content Theme | Goal Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Monday** | `08:00 - 10:00 AM` | Carousel Post | Tech Industry Trends & Infographics | Save & Share Rate |
| **Tuesday** | `01:00 - 03:00 PM` | Single Photo | Behind-the-Scenes & Team Culture | Brand Humanization |
| **Wednesday** | `05:00 - 07:00 PM` | Reel / Short Video | Quick Coding Tips & Tech Hacks | Organic Discovery |
| **Thursday** | `08:00 - 10:00 AM` | Carousel Post | Case Studies & Client Success Stories | Lead Conversion |
| **Friday** | `06:00 - 08:00 PM` | Reel / Video | Tech Quiz, Memes & Community Q&A | Comment Density |
| **Saturday** | `11:00 AM - 01:00 PM` | Photo / Story | Student Testimonials & Success Stories | Trust & Credibility |
| **Sunday** | `07:00 - 09:00 PM` | Carousel Post | Weekly Summary & Webinar Announcements | Audience Retention |

---

## 🤖 Machine Learning Model

* **Algorithm**: Random Forest Regressor (`n_estimators=100`, `max_depth=8`)
* **Target Variable**: Total Engagement (`likes_count + comments_count`)
* **Input Features**: Posting Hour, Day of Week, Is Weekend, Filter Used, Content Format, Hashtag Count, Tagged Users, Follower Count.
* **Saved Location**: `models/engagement_model.pkl`

---

## 🛠️ Quick Start & Installation

### 1. Environment Setup
Ensure you have Python 3.9+ installed, then install required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Jupyter Analysis Notebook
Open and run the pre-compiled master notebook:

```bash
jupyter notebook instagram_data_analysis.ipynb
```

### 3. Launch Streamlit Web Application
Launch the interactive web dashboard and engagement predictor:

```bash
streamlit run app.py
```

---

## 📄 License & Attribution
* **Dataset Source**: Kaggle Instagram Dataset (`bhanupratapbiswas/instgram`)
* **Project Organization**: Alfido Tech Data Science & AI Internship Program

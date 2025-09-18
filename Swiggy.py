import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import hstack, csr_matrix

# 🎨 Page Configuration
st.set_page_config(
    page_title="🍽 Swiggy Recommendations",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌸 Custom Pastel Peach Styling
st.markdown("""
<style>
body {
    background-color: #ffe5b4;
}
.main-title {
    color: #FF5722;
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}
.sidebar .sidebar-content {
    background-color: #FFF3E0;
}
.sidebar h2 {
    color: #D84315;
    font-size: 24px;
}
.restaurant-name {
    color: #BF360C;
    font-size: 26px;
    margin-bottom: 5px;
}
.stMarkdown hr {
    border: 0;
    height: 1px;
    background: #FFAB91;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# 🔄 Load Data without Caching
def load_data():
    df = pd.read_csv("E:\\guvi_projects\\Swiggy_Restaurant_Recommendation_System\\cleaned_df.csv", low_memory=True)
    return df

# 📦 Load Data & Models
df = load_data()
with open("E:\\guvi_projects\\Swiggy_Restaurant_Recommendation_System\\encoders_scaler.pkl", "rb") as f:
    encoders_scalers = pickle.load(f)
with open("E:\\guvi_projects\\Swiggy_Restaurant_Recommendation_System\\kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# 🔧 Extract Encoders/Scalers
name_encoder = encoders_scalers["name_encoder"]
cuisine_encoder = encoders_scalers["cuisine_encoder"]
city_encoder = encoders_scalers["city_encoder"]
rating_scaler = encoders_scalers["rating_scaler"]
cost_scaler = encoders_scalers["cost_scaler"]

# 🧹 Preprocessing
city_sparse = city_encoder.transform(df[["city"]])
df["name_encoded"] = name_encoder.transform(df["name"]).astype(np.float32)
df["cuisine_encoded"] = cuisine_encoder.transform(df["cuisine"]).astype(np.float32)
df["rating_scaled"] = rating_scaler.transform(df[["rating"]]).astype(np.float32)
df["cost_scaled"] = cost_scaler.transform(df[["cost"]]).astype(np.float32)

num_feats = df[["name_encoded", "cuisine_encoded", "rating_scaled", "cost_scaled"]].to_numpy(dtype=np.float32)
X = hstack([csr_matrix(num_feats), city_sparse], format="csr")

# Assign Clusters
df["cluster"] = -1
batch_size = 200_000
for start in range(0, X.shape[0], batch_size):
    end = min(start + batch_size, X.shape[0])
    df.loc[start:end, "cluster"] = kmeans.predict(X[start:end])

# 🔍 Recommendation Logic
def get_recommendations(cluster, city, cuisines, min_rating, max_cost, top_n=10):
    if "All" in cuisines:
        cuisines = df["cuisine"].unique()
    filt = df[
        (df["cluster"] == cluster) &
        (df["city"] == city) &
        (df["cuisine"].isin(cuisines)) &
        (df["rating"] >= min_rating) &
        (df["cost"] <= max_cost)
    ]
    filt = filt.sort_values(by=["rating", "cost"], ascending=[False, True])
    return filt[["name", "city", "cuisine", "cost", "rating", "rating_count", "address", "link"]].head(top_n)

# 🎯 Page Header
st.markdown("<div class='main-title'>🍽 Swiggy’s Restaurant Recommendations</div>", unsafe_allow_html=True)

# 🎛 Sidebar Input
st.sidebar.header("Your Preferences")
city = st.sidebar.selectbox("Select City", sorted(df["city"].dropna().unique()))

# ✅ Fix type error by converting to string before sorting
cuisine_options = ["All"] + sorted(df["cuisine"].dropna().astype(str).unique())
cuisines = st.sidebar.multiselect("Select Cuisines", cuisine_options, default=["All"])
rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5)

# 💰 Direct input for cost (textbox, no +/-)
cost_input = st.sidebar.text_input("Maximum Cost (INR)", "500")
try:
    cost = float(cost_input)
except ValueError:
    st.sidebar.error("Please enter a valid number for cost.")
    st.stop()

# 🔍 Prepare input for prediction
if city in city_encoder.categories_[0]:
    city_in = city_encoder.transform([[city]])
else:
    city_in = csr_matrix((1, city_sparse.shape[1]))

name_enc = -1  # Not using name in prediction
cuis_enc = cuisine_encoder.transform([cuisines[0]])[0] if cuisines and cuisines[0] != "All" else -1
user_num = np.array([[name_enc, cuis_enc,
                      rating_scaler.transform([[rating]])[0][0].astype(np.float32),
                      cost_scaler.transform([[cost]])[0][0].astype(np.float32)]], dtype=np.float32)
user_X = hstack([csr_matrix(user_num), city_in], format="csr")
user_cluster = int(kmeans.predict(user_X)[0])

# 📋 Show Recommendations
recs = get_recommendations(user_cluster, city, cuisines, rating, cost)
st.subheader("Recommended Restaurants:")
if not recs.empty:
    for _, row in recs.iterrows():
        st.markdown(f"<div class='restaurant-name'><b>{row['name']}</b></div>", unsafe_allow_html=True)
        st.write(f"Cuisine: {row['cuisine']} | Cost: ₹{row['cost']:.2f}")
        st.write(f"⭐ Rating: {row['rating']} | 🗣️ Reviews: {row['rating_count']} | 📍 {row['city']}")
        st.write(f"📌 Address: {row['address']}")
        st.write(f"🔗 Link: [View on Swiggy]({row['link']})")
        st.markdown("---")
else:
    st.warning("No restaurants match your criteria. Try adjusting filters.")

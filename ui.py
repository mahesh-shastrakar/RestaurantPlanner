import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
from math import radians, sin, cos, sqrt, atan2
import uuid
from fuzzywuzzy import fuzz

# Set page config
st.set_page_config(layout="wide", page_title="Restaurant Planner Pro", page_icon="🍽️")

# Initialize Snowflake session
try:
    session = get_active_session()
except:
    session = st.session_state.get("snowpark_session", Session.builder.getOrCreate())
    st.session_state["snowpark_session"] = session

# Default values
default_latitude = 21.1458  # Nagpur, India
default_longitude = 79.0882
default_zoom = 12

# Predefined areas with coordinates and table names
AREA_COORDINATES = {
    "Nandanvan, Nagpur": {"lat": 21.1358, "lng": 79.1223, "traffic": "Medium", "footfall": 5000, "table": "CLEANED_NANDANVAN"},
    "Chhatrapati Square, Nagpur": {"lat": 21.1092, "lng": 79.0672, "traffic": "High", "footfall": 7000, "table": "CHHATRAPATI_SQUARE_NEW"},
    "Hingana, Nagpur": {"lat": 21.0946917, "lng": 78.9748468, "traffic": "Low", "footfall": 3000, "table": "HINGANA"},
    "Sadar, Nagpur": {"lat": 21.1640, "lng": 79.0679, "traffic": "Medium", "footfall": 6000, "table": "SADAR"}
    # "Sakkardara, Nagpur": {"lat": 21.1244, "lng": 79.1082, "traffic": "Medium", "footfall": 4000, "table": "SAKKARDARA"}
}

# Initialize session state variables
if "restaurants" not in st.session_state:
    st.session_state.restaurants = pd.DataFrame()
if "selected_area" not in st.session_state:
    st.session_state.selected_area = "Nandanvan, Nagpur"
if "marker_location" not in st.session_state:
    st.session_state.marker_location = [default_latitude, default_longitude]
if "clustered" not in st.session_state:
    st.session_state.clustered = False
if "selected_cuisines" not in st.session_state:
    st.session_state.selected_cuisines = []
if "all_cuisines" not in st.session_state:
    st.session_state.all_cuisines = []
if "selected_table" not in st.session_state:
    st.session_state.selected_table = "CLEANED_NANDANVAN"
if "radius" not in st.session_state:
    st.session_state.radius = 5
if "show_output" not in st.session_state:
    st.session_state.show_output = False
if "suggestion" not in st.session_state:
    st.session_state.suggestion = ""
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = {}

# Haversine formula to calculate distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Find closest predefined area
def find_closest_area(lat, lng, threshold_km=1.0):
    for area, coords in AREA_COORDINATES.items():
        distance = haversine_distance(lat, lng, coords["lat"], coords["lng"])
        if distance <= threshold_km:
            return area, coords
    return None, None

# Calculate cluster centroid and radius
def calculate_cluster_centroid_and_radius(df, cluster_name):
    cluster_df = df[df['cluster_name'] == cluster_name]
    if cluster_df.empty:
        return None, None, None
    centroid_lat = cluster_df['latitude'].mean()
    centroid_lon = cluster_df['longitude'].mean()
    max_distance = cluster_df.apply(
        lambda row: haversine_distance(centroid_lat, centroid_lon, row['latitude'], row['longitude']),
        axis=1
    ).max() if len(cluster_df) > 1 else 0.5
    return centroid_lat, centroid_lon, max_distance

# Standardize price level
def standardize_price_level(price):
    try:
        price = float(price)
        if price <= 200:
            return np.random.choice([1, 2], p=[0.8, 0.2])
        elif price <= 500:
            return np.random.choice([1, 2, 3], p=[0.1, 0.7, 0.2])
        elif price <= 1000:
            return np.random.choice([2, 3, 4], p=[0.1, 0.8, 0.1])
        else:
            return np.random.choice([3, 4], p=[0.3, 0.7])
    except:
        return np.random.choice([1, 2, 3, 4], p=[0.2, 0.5, 0.2, 0.1])

# Calculate dynamic price range
def calculate_dynamic_price_range(df):
    price_ranges = {1: (0, 200), 2: (201, 500), 3: (501, 1000), 4: (1000, 2000)}
    if len(df) >= 3:
        df['PRICE_LEVEL'] = np.random.choice([1, 2, 3, 4], size=len(df), p=[0.4, 0.4, 0.1, 0.1])
    else:
        df['PRICE_LEVEL'] = np.random.randint(1, 5, size=len(df))
    price_level_avg = {
        1: np.random.beta(2, 1) * 200,
        2: 200 + np.random.beta(2, 1) * 300,
        3: 500 + np.random.beta(2, 1) * 500,
        4: 1000 + np.random.beta(2, 1) * 500
    }
    df['PRICE_RANGE'] = df['PRICE_LEVEL'].map(price_level_avg)
    if len(df) >= 3 and 3 not in df['PRICE_LEVEL'].unique():
        df.iloc[-1, df.columns.get_loc('PRICE_LEVEL')] = 3
        df.iloc[-1, df.columns.get_loc('PRICE_RANGE')] = price_level_avg[3]
    return df

# Load data from Snowflake
@st.cache_data
def load_data(table_name):
    try:
        query = f"SELECT * FROM RESTAURANTDB.RESTAURANT_SCHEMA.{table_name}"
        df = session.sql(query).to_pandas()
        df['CUISINETYPE'] = df['CUISINETYPE'].fillna('')
        all_cuisines = set()
        for cuisines in df['CUISINETYPE'].str.split(','):
            if isinstance(cuisines, list):
                all_cuisines.update([c.strip() for c in cuisines if c.strip()])
        df['ALL_CUISINES'] = df['CUISINETYPE'].str.split(',').apply(
            lambda x: [c.strip() for c in x if c.strip()] if isinstance(x, list) else []
        )
        df['PRICE_LEVEL'] = df['PRICE_LEVEL'].apply(standardize_price_level)
        df['PRICE_LEVEL'] = df['PRICE_LEVEL'].apply(
            lambda x: x if pd.notna(x) and 1 <= x <= 4 else np.random.randint(1, 5)
        )
        df = calculate_dynamic_price_range(df)
        return df, sorted(all_cuisines)
    except Exception as e:
        st.error(f"Error loading data from {table_name}: {str(e)}")
        return pd.DataFrame(), []

# Fetch places data
def fetch_places_data(lat, lng, radius_km, table_name, cuisines=None):
    df, _ = load_data(table_name)
    if not df.empty:
        df['distance'] = df.apply(
            lambda row: haversine_distance(lat, lng, row['LAT'], row['LNG']),
            axis=1
        )
        df = df[df['distance'] <= radius_km]
        if cuisines:
            df = df[df['ALL_CUISINES'].apply(lambda x: any(fuzz.token_sort_ratio(c.strip().lower(), cuisine.strip().lower()) >= 80 for cuisine in x for c in cuisines))]
        return df.rename(columns={
            'LAT': 'latitude',
            'LNG': 'longitude',
            'NAME': 'name',
            'CUISINETYPE': 'cuisine',
            'PRICE_LEVEL': 'price_level',
            'PRICE_RANGE': 'price_range',
            'RATING': 'rating',
            'USER_RATINGS_TOTAL': 'footfall',
            'ALL_CUISINES': 'all_cuisines'
        })[['latitude', 'longitude', 'name', 'cuisine', 'price_level', 'price_range', 'rating', 'footfall', 'all_cuisines']]
    else:
        st.warning(f"No data available from {table_name} table. Using mock data.")
        mock_data = [
            {"name": "Restaurant 1", "latitude": lat + 0.01, "longitude": lng + 0.01, 
             "cuisine": "Italian, French", "all_cuisines": ["Italian", "French"], 
             "price_level": np.random.choice([1, 2], p=[0.8, 0.2]), "price_range": 175, "rating": 4.5, "footfall": 150},
            {"name": "Restaurant 2", "latitude": lat - 0.01, "longitude": lng - 0.01, 
             "cuisine": "Chinese, Thai", "all_cuisines": ["Chinese", "Thai"], 
             "price_level": np.random.choice([1, 2, 3], p=[0.1, 0.7, 0.2]), "price_range": 400, "rating": 4.0, "footfall": 200},
            {"name": "Restaurant 3", "latitude": lat + 0.02, "longitude": lng - 0.02, 
             "cuisine": "Indian, Chinese", "all_cuisines": ["Indian", "Chinese"], 
             "price_level": np.random.choice([2, 3, 4], p=[0.2, 0.7, 0.1]), "price_range": 800, "rating": 4.2, "footfall": 180},
            {"name": "Restaurant 4", "latitude": lat + 0.015, "longitude": lng - 0.015, 
             "cuisine": "Mexican, American", "all_cuisines": ["Mexican", "American"], 
             "price_level": np.random.choice([3, 4], p=[0.3, 0.7]), "price_range": 1500, "rating": 4.1, "footfall": 170}
        ]
        mock_df = pd.DataFrame(mock_data)
        mock_df['distance'] = mock_df.apply(
            lambda row: haversine_distance(lat, lng, row['latitude'], row['longitude']),
            axis=1
        )
        return mock_df[mock_df['distance'] <= radius_km][['latitude', 'longitude', 'name', 'cuisine', 'price_level', 'price_range', 'rating', 'footfall', 'all_cuisines']]

# Perform clustering
def perform_clustering(df, selected_cuisines=None):
    if df.empty or len(df) < 2:
        st.warning("Not enough data points for clustering (need at least 2)")
        return df
    if selected_cuisines:
        df = df[df['all_cuisines'].apply(lambda x: any(fuzz.token_sort_ratio(c.strip().lower(), cuisine.strip().lower()) >= 80 for cuisine in x for c in selected_cuisines))]
    if df.empty:
        st.warning("No restaurants match the selected cuisines. Proceeding with all data.")
        df = st.session_state.restaurants
    features = df[['latitude', 'longitude']].copy()
    for col in ['latitude', 'longitude']:
        min_val = features[col].min()
        max_val = features[col].max()
        if max_val > min_val:
            features[col] = (features[col] - min_val) / (max_val - min_val)
        else:
            features[col] = 0
    max_clusters = min(10, len(df))
    distortions = []
    K = range(1, max_clusters + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        distortions.append(kmeans.inertia_)
    differences = np.diff(distortions)
    elbow_point = next((i for i, diff in enumerate(differences[:-1]) 
                        if i + 1 < len(differences) and diff / differences[i + 1] > 2), max_clusters - 1)
    n_clusters = max(5, elbow_point + 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(features)
    cluster_to_area = {cluster: f"Area {cluster}" for cluster in df['cluster'].unique()}
    df['cluster_name'] = df['cluster'].map(cluster_to_area)
    cluster_stats = df.groupby('cluster')['footfall'].agg(['mean', 'std']).fillna(0)
    global_mean = df['footfall'].mean()
    traffic_classes = []
    for cluster in df['cluster'].unique():
        cluster_mean = cluster_stats.loc[cluster, 'mean']
        if cluster_mean > global_mean * 1.2:
            traffic_classes.append(('BUSY', cluster))
        elif cluster_mean < global_mean * 0.8:
            traffic_classes.append(('NON BUSY', cluster))
        else:
            traffic_classes.append(('MODERATE', cluster))
    traffic_map = {cluster: traffic for traffic, cluster in traffic_classes}
    df['traffic_class'] = df['cluster'].map(traffic_map)
    return df

# Analyze cuisines with fuzzy matching
def analyze_cuisines(restaurants_df, selected_cuisines, selected_area, lat, lng, radius_km):
    if restaurants_df.empty or 'all_cuisines' not in restaurants_df.columns:
        return []
    if selected_area in AREA_COORDINATES:
        restaurants_df = restaurants_df[restaurants_df.apply(
            lambda row: haversine_distance(lat, lng, row['latitude'], row['longitude']) <= radius_km,
            axis=1
        )]
    cuisine_stats = []
    cuisines_to_analyze = selected_cuisines if selected_cuisines else st.session_state.all_cuisines
    for cuisine in cuisines_to_analyze:
        subset = restaurants_df[restaurants_df['all_cuisines'].apply(
            lambda x: any(fuzz.token_sort_ratio(cuisine.strip().lower(), c.strip().lower()) >= 80 for c in x) if isinstance(x, list) else False
        )]
        if not subset.empty:
            cuisine_stats.append({
                "cuisine": cuisine,
                "avg_rating": subset['rating'].mean(),
                "avg_price": subset['price_range'].mean() if subset['price_range'].notna().any() else 0,
                "competitors": len(subset),
                "selected": cuisine in selected_cuisines
            })
    return sorted(cuisine_stats, key=lambda x: x["avg_rating"], reverse=True)

# Get top cuisine for selected area
def get_top_cuisine_for_selected_area(df, selected_area, lat, lng, radius_km, selected_cuisines):
    if df.empty or 'all_cuisines' not in df.columns:
        return []
    if selected_area in AREA_COORDINATES:
        area_df = df[df['cluster_name'] == f"Area {df['cluster'].iloc[0]}" if not df.empty else selected_area]
        area_name = selected_area
    else:
        area_df = df[df.apply(
            lambda row: haversine_distance(lat, lng, row['latitude'], row['longitude']) <= radius_km,
            axis=1
        )]
        area_name = selected_area if selected_area != "Custom Location" else "Selected Area"
    if area_df.empty:
        return []
    cuisine_ratings = []
    cuisines_to_analyze = selected_cuisines if selected_cuisines else st.session_state.all_cuisines
    for cuisine in cuisines_to_analyze:
        subset = area_df[area_df['all_cuisines'].apply(
            lambda x: any(fuzz.token_sort_ratio(cuisine.strip().lower(), c.strip().lower()) >= 80 for c in x) if isinstance(x, list) else False
        )]
        if not subset.empty:
            cuisine_ratings.append({
                "cuisine": cuisine,
                "avg_rating": subset['rating'].mean(),
                "avg_price": subset['price_range'].mean() if subset['price_range'].notna().any() else 0,
                "competitors": len(subset),
                "area_name": area_name
            })
    if not cuisine_ratings:
        return []
    cuisine_ratings = sorted(cuisine_ratings, key=lambda x: x["avg_rating"], reverse=True)
    top_rating = cuisine_ratings[0]["avg_rating"]
    return [x for x in cuisine_ratings if abs(x["avg_rating"] - top_rating) <= 0.1]

# Analyze clusters
def analyze_clusters(restaurants_df, selected_cuisines):
    analysis = {}
    if 'cluster' not in restaurants_df.columns:
        return {'cluster_stats': pd.DataFrame(), 'traffic_stats': pd.DataFrame()}
    if selected_cuisines:
        cuisine_df = restaurants_df[restaurants_df['all_cuisines'].apply(
            lambda x: any(fuzz.token_sort_ratio(c.strip().lower(), cuisine.strip().lower()) >= 80 for cuisine in x for c in selected_cuisines)
        )]
    else:
        cuisine_df = restaurants_df
    try:
        cluster_stats = cuisine_df.groupby(['cluster', 'cluster_name']).agg({
            'rating': 'mean',
            'price_range': ['mean', 'count'],
            'footfall': 'mean'
        }).reset_index()
        cluster_stats.columns = ['cluster', 'cluster_name', 'avg_rating', 'avg_price', 'count', 'avg_footfall']
        all_clusters = set(restaurants_df['cluster'].unique())
        missing_clusters = all_clusters - set(cluster_stats['cluster'])
        if missing_clusters:
            missing_data = pd.DataFrame([{
                'cluster': cluster,
                'cluster_name': restaurants_df[restaurants_df['cluster'] == cluster]['cluster_name'].iloc[0] if not restaurants_df[restaurants_df['cluster'] == cluster].empty else f"Area {cluster}",
                'avg_rating': 0,
                'avg_price': 0,
                'count': 0,
                'avg_footfall': 0
            } for cluster in missing_clusters])
            cluster_stats = pd.concat([cluster_stats, missing_data], ignore_index=True)
        analysis['cluster_stats'] = cluster_stats
    except Exception as e:
        st.warning(f"Could not analyze clusters: {str(e)}")
        analysis['cluster_stats'] = pd.DataFrame()
    try:
        if 'traffic_class' in cuisine_df.columns:
            traffic_stats = cuisine_df.groupby('traffic_class').agg({
                'rating': 'mean',
                'price_range': 'mean',
                'footfall': 'mean'
            }).reset_index()
            analysis['traffic_stats'] = traffic_stats
        else:
            analysis['traffic_stats'] = pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not analyze traffic classes: {str(e)}")
        analysis['traffic_stats'] = pd.DataFrame()
    return analysis

# Calculate density
def calculate_density(restaurants_df, radius_km):
    area_sq_km = np.pi * (radius_km ** 2)
    total_restaurants = len(restaurants_df)
    return total_restaurants / area_sq_km if area_sq_km > 0 else 0

# Calculate average spending
def calculate_avg_spending(df):
    if df.empty or 'price_range' not in df.columns:
        return 0
    return df['price_range'].mean()

# Create circle points for map
def create_circle_points(center_lat, center_lng, radius_km, num_points=100):
    radius_deg = radius_km / 111
    angles = np.linspace(0, 2 * np.pi, num_points)
    return pd.DataFrame({
        'lat': center_lat + radius_deg * np.cos(angles),
        'lon': center_lng + radius_deg * np.sin(angles),
        'cluster': [-2] * num_points,
        'cluster_name': ['Cluster Boundary'] * num_points,
        'color': ['#0000FF'] * num_points
    })

# Get quick suggestion
def get_quick_suggestion(df, selected_cuisines):
    if df.empty:
        return "No data available to provide a suggestion.", None, None, None, None, None
    if not selected_cuisines:
        return "Please select at least one cuisine to generate a suggestion.", None, None, None, None, None
    clustered_df = perform_clustering(df, selected_cuisines)
    if clustered_df.empty or 'cluster' not in clustered_df.columns:
        return "No data available after clustering to provide a suggestion.", None, None, None, None, None
    # Perform predictive modeling to get top-scoring cluster
    try:
        cluster_stats = clustered_df.groupby(['cluster', 'cluster_name']).agg({
            'rating': 'mean',
            'footfall': 'mean',
            'price_range': 'mean'
        }).reset_index()
        rating_weight = 0.4
        traffic_weight = 0.4
        price_weight = 0.2
        max_rating = cluster_stats['rating'].max()
        max_footfall = cluster_stats['footfall'].max()
        # Calculate price score: prioritize low (≤500) and medium (501-1000) prices
        cluster_stats['price_score'] = cluster_stats['price_range'].apply(
            lambda x: 1.0 if x <= 500 else 0.8 if x <= 1000 else 0.2
        )
        cluster_stats['score'] = (
            (rating_weight * (cluster_stats['rating'] / max_rating if max_rating > 0 else 0)) +
            (traffic_weight * (cluster_stats['footfall'] / max_footfall if max_footfall > 0 else 0)) +
            (price_weight * cluster_stats['price_score'])
        )
        top_cluster = cluster_stats.sort_values('score', ascending=False).iloc[0]
        cluster_name = top_cluster['cluster_name']
        avg_rating = top_cluster['rating']
        avg_price = top_cluster['price_range']
        footfall = top_cluster['footfall']
        competitors = len(clustered_df[clustered_df['cluster_name'] == cluster_name])
        centroid_lat, centroid_lon, cluster_radius = calculate_cluster_centroid_and_radius(clustered_df, cluster_name)
        # Analyze combined cuisines
        cuisine_subset = clustered_df[clustered_df['all_cuisines'].apply(
            lambda x: any(fuzz.token_sort_ratio(c.strip().lower(), cuisine.strip().lower()) >= 80 for cuisine in x for c in selected_cuisines)
        )]
        if cuisine_subset.empty:
            return "No restaurants match the selected cuisines in this area.", None, None, None, None, None
        combined_rating = cuisine_subset['rating'].mean()
        combined_price = cuisine_subset['price_range'].mean()
        combined_competitors = len(cuisine_subset)
        cuisine_label = " , ".join(selected_cuisines) if len(selected_cuisines) > 1 else selected_cuisines[0]
        reasons = []
        if footfall > 150:
            reasons.append("steady foot traffic")
        if avg_rating > 4.0:
            reasons.append("high customer satisfaction")
        if not reasons:
            reasons.append("good market potential")
        reason_text = " and ".join(reasons)
        density = calculate_density(cuisine_subset, st.session_state.radius)
        rating_color = 'green' if avg_rating >= 4.0 else 'yellow' if avg_rating >= 3.0 else 'red'
        rating_status = 'High' if avg_rating >= 4.0 else 'Moderate' if avg_rating >= 3.0 else 'Low'
        price_color = 'green' if avg_price <= 500 else 'yellow' if avg_price <= 1000 else 'red'
        price_status = 'Affordable' if avg_price <= 500 else 'Moderate' if avg_price <= 1000 else 'Premium'
        density_color = 'green' if density <= 5 else 'yellow' if density <= 10 else 'red'
        density_status = 'Low' if density <= 5 else 'Moderate' if density <= 10 else 'High'
        suggestion = f"""
# Best Area to Open Your **{cuisine_label}** Spot

**Summary**: **{cluster_name}** is ideal for your restaurant with **{reason_text}**.

## 🗺️ Recommended Area: {cluster_name}

### Why This Area?
| Metric | Value | Status | Comparison |
|--------|-------|--------|------------|
| ⭐ **Rating** | {avg_rating:.1f}/5 | <span style="color:{rating_color}">{rating_status}</span> | <progress value="{avg_rating * 20}" max="100"></progress> |
| 💰 **Avg Price** | ₹{int(avg_price)} | <span style="color:{price_color}">{price_status}</span> | <progress value="{min(avg_price / 20, 100)}" max="100"></progress> |
| 📍 **Density** | {density:.1f}/km² | <span style="color:{density_color}">{density_status}</span> | <progress value="{min(density * 10, 100)}" max="100"></progress> |

### What's Great & What's Not🤔
**Pros**:
- High customer ratings ({avg_rating:.1f} ⭐)
- {reason_text.capitalize()}
- Balanced pricing (₹{int(avg_price)} avg)

**Cons**:
- Moderate competition ({competitors} restaurants)

### Why It Stands Out?
This area offers **{reason_text}**, making it a prime spot for your **{cuisine_label}** spot. Explore properties or dive deeper with a full analysis!

**Ready to explore {cluster_name}?** Click **Analyze Area** for detailed insights!
"""
        return suggestion, cluster_name, clustered_df, centroid_lat, centroid_lon, cluster_radius
    except Exception as e:
        return f"Error generating suggestion: {str(e)}", None, None, None, None, None

# Predictive modeling
def show_predictive_modeling(df):
    st.subheader("🔮 Prediction")
    st.write("Predict the best locations for your restaurant concept based on historical data and market trends.")
    if df.empty:
        st.warning("Please load restaurant data first by analyzing an area.")
        return
    if not st.session_state.selected_cuisines:
        st.warning("Please select cuisines in Step 2 to enable predictive modeling")
        return
    cuisine_label = " and ".join(st.session_state.selected_cuisines) if len(st.session_state.selected_cuisines) > 1 else st.session_state.selected_cuisines[0]
    with st.spinner(f"Training model for {cuisine_label} cuisine..."):
        try:
            cuisine_df = perform_clustering(df, st.session_state.selected_cuisines)
            if cuisine_df.empty or 'cluster' not in cuisine_df.columns:
                st.error(f"No data available for {cuisine_label} after clustering")
                return
            cluster_stats = cuisine_df.groupby(['cluster', 'cluster_name']).agg({
                'rating': 'mean',
                'footfall': 'mean',
                'price_range': 'mean'
            }).reset_index()
            rating_weight = 0.4
            traffic_weight = 0.4
            price_weight = 0.2
            max_rating = cluster_stats['rating'].max()
            max_footfall = cluster_stats['footfall'].max()
            # Calculate price score: prioritize low (≤500) and medium (501-1000) prices
            cluster_stats['price_score'] = cluster_stats['price_range'].apply(
                lambda x: 1.0 if x <= 500 else 0.8 if x <= 1000 else 0.2
            )
            cluster_stats['score'] = (
                (rating_weight * (cluster_stats['rating'] / max_rating if max_rating > 0 else 0)) +
                (traffic_weight * (cluster_stats['footfall'] / max_footfall if max_footfall > 0 else 0)) +
                (price_weight * cluster_stats['price_score'])
            )
            top_clusters = cluster_stats.sort_values('score', ascending=False).head(3)
            st.write(f"### Top Recommended Areas for {cuisine_label}")
            cols = st.columns(3)
            for idx, (_, row) in enumerate(top_clusters.iterrows()):
                with cols[idx]:
                    st.metric(
                        label=f"{row['cluster_name']}",
                        value=f"Score: {row['score']:.2f}",
                        delta=f"{len(cuisine_df[cuisine_df['cluster']==row['cluster']])} restaurants"
                    )
                    st.write(f"⭐ Avg Rating: {row['rating']:.1f}")
                    st.write(f"👥 Avg Footfall: {int(row['footfall'])}")
                    st.write(f"💰 Avg Price: ₹{row['price_range']:.0f}")
            st.write("### Cluster Map")
            map_df = cuisine_df[['latitude', 'longitude', 'cluster', 'cluster_name']].copy()
            map_df = map_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
            cluster_colors = {
                0: "#FF0000", 1: "#00FF00", 2: "#FFFF00", 3: "#FF00FF", 4: "#00FFFF",
                5: "#800080", 6: "#FFA500", 7: "#008000", 8: "#FFC0CB", 9: "#A52A2A"
            }
            map_df['color'] = map_df['cluster'].map(cluster_colors)
            circle_df = create_circle_points(
                st.session_state.marker_location[0],
                st.session_state.marker_location[1],
                radius_km=5
            )
            marker_df = pd.DataFrame({
                'lat': [st.session_state.marker_location[0]],
                'lon': [st.session_state.marker_location[1]],
                'cluster': [-1],
                'cluster_name': ['Selected Location'],
                'color': ["#000000"]
            })
            combined_df = pd.concat([map_df, marker_df, circle_df], ignore_index=True)
            st.map(
                combined_df,
                latitude='lat',
                longitude='lon',
                color='color',
                size=15,
                zoom=12,
                use_container_width=True
            )
            st.write("**Area Legend**")
            for cluster, color in cluster_colors.items():
                if cluster in map_df['cluster'].unique():
                    cluster_name = cuisine_df[cuisine_df['cluster'] == cluster]['cluster_name'].iloc[0]
                    st.markdown(f"<span style='color:{color}'>■</span> {cluster_name}", unsafe_allow_html=True)
            st.markdown("<span style='color:#000000'>■</span> Selected Location", unsafe_allow_html=True)
            st.markdown("<span style='color:#0000FF'>■</span> Search Radius", unsafe_allow_html=True)
            top_cluster = top_clusters['cluster'].iloc[0]
            sample_restaurants = cuisine_df[cuisine_df['cluster'] == top_cluster].sample(min(3, len(cuisine_df)))
            st.write("### Sample Restaurants in Top Areas")
            for _, restaurant in sample_restaurants.iterrows():
                with st.expander(f"🍽️ {restaurant['name']}"):
                    st.write(f"**Rating**: {restaurant['rating']} ⭐")
                    st.write(f"**Footfall**: {restaurant['footfall']} reviews")
                    st.write(f"**Price Level**: {restaurant['price_level']}")
                    st.write(f"**Price Range**: ₹{restaurant['price_range']:.0f}")
                    st.write(f"**Cuisines**: {', '.join(restaurant['all_cuisines'])}")
        except Exception as e:
            st.error(f"Error in predictive modeling: {str(e)}")

# Budget analysis
def show_budget_analysis(df, selected_area):
    st.subheader("💰 Budget Analysis")
    st.write("Analyze price competitiveness and customer spending patterns across distinct price levels.")
    if df.empty:
        st.warning("Please load restaurant data first by analyzing an area.")
        return
    if 'price_level' not in df.columns:
        st.warning("No price data available for budget analysis")
        return
    def map_price_level(price):
        try:
            price = int(price)
            if price == 1:
                return 'Budget (≤₹200)'
            elif price == 2:
                return 'Moderate (₹201-500)'
            elif price == 3:
                return 'Premium (₹501-1000)'
            elif price == 4:
                return 'Luxury (>₹1000)'
            else:
                return 'Unknown'
        except:
            return 'Unknown'
    df['price_category'] = df['price_level'].apply(map_price_level)
    all_categories = ['Budget (≤₹200)', 'Moderate (₹201-500)', 'Premium (₹501-1000)', 'Luxury (>₹1000)']
    price_stats = df.groupby('price_category').agg({
        'rating': 'mean',
        'footfall': 'mean',
        'price_level': 'count'
    }).rename(columns={'price_level': 'count'}).reset_index()
    missing_categories = [cat for cat in all_categories if cat not in price_stats['price_category'].values]
    if missing_categories:
        missing_data = pd.DataFrame({
            'price_category': missing_categories,
            'rating': [0] * len(missing_categories),
            'footfall': [0] * len(missing_categories),
            'count': [0] * len(missing_categories)
        })
        price_stats = pd.concat([price_stats, missing_data], ignore_index=True)
    price_stats['sort_order'] = price_stats['price_category'].apply(lambda x: all_categories.index(x) if x in all_categories else len(all_categories))
    price_stats = price_stats.sort_values('sort_order').drop(columns='sort_order')
    avg_spending = calculate_avg_spending(df)
    st.write("### 📌 Recommendations")
    if len(df) >= 10:
        dominant_category = price_stats.loc[price_stats['count'].idxmax(), 'price_category']
        dominant_count = price_stats['count'].max()
        st.write(f"**Dominant Price Segment**: {dominant_category} ({dominant_count} restaurants)")
        st.write("")
        if dominant_category == 'Budget (≤₹200)':
            st.write("**Strategy**: Focus on high-volume, low-cost offerings with quick service.")
            st.write("**Differentiation**: Offer unique flavors or promotions to stand out.")
        elif dominant_category == 'Moderate (₹201-500)':
            st.write("**Strategy**: Target mid-range diners with a balance of quality and affordability.")
            st.write("**Differentiation**: Introduce premium elements (e.g., ambiance, specialty dishes) at moderate prices.")
        elif dominant_category == 'Premium (₹501-1000)':
            st.write("**Strategy**: Invest in quality ingredients and dining experience to attract premium customers.")
            st.write("**Differentiation**: Offer exclusive menu items or loyalty programs.")
        else:
            st.write("**Strategy**: Create a high-end dining experience with premium branding.")
            st.write("**Differentiation**: Focus on exclusivity, fine dining, or unique culinary concepts.")
    else:
        st.warning("Limited data available. Consider expanding the analysis radius for more accurate recommendations.")
    st.write("### 💡 Budget Insights")
    st.write(f"**Average Spending**: ₹{avg_spending:.0f}")
    st.write("")
    st.write("**Price Level Definitions**:")
    st.write("")
    st.write("- [0-200₹] - Budget (1)")
    st.write("- [201-500₹] - Moderate (2)")
    st.write("- [501-1000₹] - Premium (3)")
    st.write("- [1000₹+] - Luxury (4)")
    st.write("")
    if avg_spending <= 200:
        st.write("This area is budget-conscious. Customers prioritize affordability.")
    elif avg_spending <= 500:
        st.write("This area supports moderate pricing. Balance quality and cost.")
    elif avg_spending <= 1000:
        st.write("This area leans toward premium dining. Focus on quality and experience.")
    else:
        st.write("This is a luxury market. Customers expect high-end dining experiences.")
    st.write("### 🏆 Competitive Price Positioning")
    st.dataframe(
        price_stats[['price_category', 'rating', 'footfall', 'count']].style.format({
            'rating': '{:.2f}',
            'footfall': '{:.2f}',
            'count': '{:.0f}'
        }).highlight_max(subset=['rating', 'footfall', 'count'], color='darkgreen')
        .highlight_min(subset=['rating', 'footfall', 'count'], color='lightcoral')
    )
    st.write("### 📊 Price Category Distribution")
    fig = px.bar(
        price_stats,
        x='price_category',
        y='count',
        title='Number of Restaurants by Price Category',
        labels={'count': 'Number of Restaurants', 'price_category': 'Price Category'},
        height=400
    )
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

# Analysis dashboard
def show_analysis_dashboard(df, selected_area, radius, selected_cuisines):
    tab1, tab2, tab3 = st.tabs(["📊 Market Analysis", "🔮 Prediction", "💰 Budget Analysis"])
    with tab1:
        st.subheader("📊 Market Analysis Dashboard")
        clustered_df = perform_clustering(df, selected_cuisines)
        st.session_state.restaurants = clustered_df
        st.session_state.clustered = True
        st.write("### 💡 Best Area to Open Your Restaurant")
        analysis = analyze_clusters(clustered_df, selected_cuisines)
        if not analysis['cluster_stats'].empty:
            cluster_stats = analysis['cluster_stats']
            selected_area_stats = cluster_stats[cluster_stats['cluster_name'] == f"Area {clustered_df['cluster'].iloc[0]}" if not clustered_df.empty else selected_area]
            if not selected_area_stats.empty and selected_area_stats['count'].iloc[0] > 0:
                best_cluster = selected_area_stats.iloc[0]
            else:
                best_cluster = cluster_stats[cluster_stats['count'] > 0].sort_values('avg_rating', ascending=False).iloc[0]
                st.warning(f"No sufficient data for {selected_area}. Showing the next best area instead.")
            if selected_cuisines:
                st.info(f"For your selected cuisines, consider locations in this area with differentiating factors.")
        else:
            st.warning("Insufficient data to generate specific recommendations")
        col1, col2, col3, col4 = st.columns(4)
        density = calculate_density(clustered_df, radius)
        col1.metric("Restaurant Density", f"{density:.1f}/km²", help="Number of restaurants in this area per km²")
        col2.metric("Average Rating", f"{clustered_df['rating'].mean():.1f} ⭐" if not clustered_df.empty else "N/A",
                    help=f"Based on {int(clustered_df['footfall'].sum())} ratings" if not clustered_df.empty else "N/A")
        col3.metric("Average Price", f"₹{calculate_avg_spending(clustered_df):.0f}" if not clustered_df.empty and clustered_df['price_range'].notna().any() else "N/A",
                    help="Typical spending per meal")
        col4.metric("Total Competitors", len(clustered_df))
        st.write("### 🌆 Area Insights")
        if selected_area in AREA_COORDINATES:
            area_data = AREA_COORDINATES[selected_area]
            col1, col2, col3 = st.columns(3)
            col1.metric("Traffic Level", area_data['traffic'])
            col2.metric("Daily Customers", f"{area_data['footfall']:,}", help="Estimated daily diners")
            col3.metric("Avg Price", f"₹{calculate_avg_spending(clustered_df):.0f}")
        else:
            st.warning("Custom location selected - area insights unavailable")
        st.write("### 🍜 Cuisine Analysis")
        if selected_cuisines:
            st.write(f"**Selected Cuisines**: {', '.join(selected_cuisines)}")
        else:
            st.write("**No specific cuisines selected** - analyzing all cuisines")
        cuisine_stats = analyze_cuisines(
            clustered_df,
            selected_cuisines,
            selected_area,
            st.session_state.marker_location[0],
            st.session_state.marker_location[1],
            radius
        )
        if cuisine_stats:
            st.write("#### Top Performing Cuisine in Selected Area")
            top_cuisines = get_top_cuisine_for_selected_area(
                clustered_df,
                selected_area,
                st.session_state.marker_location[0],
                st.session_state.marker_location[1],
                radius,
                selected_cuisines
            )
            if top_cuisines:
                cols = st.columns(min(len(top_cuisines), 3))
                for idx, cuisine_data in enumerate(top_cuisines):
                    with cols[idx % 3]:
                        st.metric(
                            label=f"🏆 {cuisine_data['cuisine']}",
                            value=f"{cuisine_data['avg_rating']:.1f} ⭐",
                            delta=f"Price: ₹{cuisine_data['avg_price']:.0f}" if cuisine_data['avg_price'] > 0 else "N/A"
                        )
            else:
                st.warning(f"No data for selected cuisines ({', '.join(selected_cuisines)}) in {selected_area}.")
            if selected_cuisines:
                st.write("#### Your Selected Cuisines Performance")
                selected_stats = [stat for stat in cuisine_stats if stat['cuisine'].lower() in [c.lower() for c in selected_cuisines]]
                if selected_stats:
                    num_cols = min(len(selected_stats), 3)
                    cols = st.columns(num_cols)
                    for idx, cuisine_data in enumerate(selected_stats):
                        with cols[idx % num_cols]:
                            if cuisine_data['competitors'] == 0:
                                st.warning(f"No data for {cuisine_data['cuisine']} in this area.")
                            else:
                                st.metric(
                                    label=f"🔹 {cuisine_data['cuisine']}",
                                    value=f"{cuisine_data['avg_rating']:.1f} ⭐",
                                    delta=f"{cuisine_data['competitors']} competitors"
                                )
                                st.write(f"**Avg Price**: ₹{cuisine_data['avg_price']:.0f}")
                else:
                    st.warning("No performance data available for the selected cuisines.")
        else:
            st.warning("Please select cuisines for analysis")
        st.write("### 📍 Area Analysis")
        subtab1, subtab2 = st.tabs(["Area Metrics", "Traffic Class Metrics"])
        with subtab1:
            if not analysis['cluster_stats'].empty:
                cluster_stats = analysis['cluster_stats']
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        "Average Rating by Area",
                        "Average Price by Area",
                        "Restaurant Count by Area",
                        "Average Footfall by Area"
                    ]
                )
                fig.add_trace(
                    go.Bar(x=cluster_stats['cluster_name'], y=cluster_stats['avg_rating'], name='Rating'),
                    row=1, col=1
                )
                fig.update_xaxes(title_text="Area", row=1, col=1, tickangle=0)
                fig.update_yaxes(title_text="Average Rating", row=1, col=1)
                fig.add_trace(
                    go.Bar(x=cluster_stats['cluster_name'], y=cluster_stats['avg_price'], name='Price'),
                    row=1, col=2
                )
                fig.update_xaxes(title_text="Area", row=1, col=2, tickangle=0)
                fig.update_yaxes(title_text="Average Price (₹)", row=1, col=2)
                fig.add_trace(
                    go.Bar(x=cluster_stats['cluster_name'], y=cluster_stats['count'], name='Count'),
                    row=2, col=1
                )
                fig.update_xaxes(title_text="Area", row=2, col=1, tickangle=0)
                fig.update_yaxes(title_text="Number of Restaurants", row=2, col=1)
                fig.add_trace(
                    go.Bar(x=cluster_stats['cluster_name'], y=cluster_stats['avg_footfall'], name='Customers'),
                    row=2, col=2
                )
                fig.update_xaxes(title_text="Area", row=2, col=2, tickangle=0)
                fig.update_yaxes(title_text="Average No. of Customers", row=2, col=2)
                fig.update_layout(height=700, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No cluster data available. Please perform clustering first.")
        with subtab2:
            if not analysis['traffic_stats'].empty:
                fig = px.bar(
                    analysis['traffic_stats'],
                    x='traffic_class',
                    y=['rating', 'price_range', 'footfall'],
                    barmode='group',
                    title='Metrics by Traffic Class',
                    labels={'value': 'Metric Value', 'variable': 'Metric'},
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No traffic class data available. Please perform clustering first.")
        st.write("### 🗺️ Interactive Map")
        if not clustered_df.empty:
            map_df = clustered_df[['latitude', 'longitude', 'cluster', 'cluster_name']].copy()
            map_df = map_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
            cluster_colors = {
                0: "#FF0000", 1: "#00FF00", 2: "#FFFF00", 3: "#FF00FF", 4: "#00FFFF",
                5: "#800080", 6: "#FFA500", 7: "#008000", 8: "#FFC0CB", 9: "#A52A2A"
            }
            map_df['color'] = map_df['cluster'].map(cluster_colors)
            circle_df = create_circle_points(
                st.session_state.marker_location[0],
                st.session_state.marker_location[1],
                radius
            )
            marker_df = pd.DataFrame({
                'lat': [st.session_state.marker_location[0]],
                'lon': [st.session_state.marker_location[1]],
                'cluster': [-1],
                'cluster_name': ['Selected Location'],
                'color': ["#000000"]
            })
            combined_df = pd.concat([map_df, marker_df, circle_df], ignore_index=True)
            st.map(
                combined_df,
                latitude='lat',
                longitude='lon',
                color='color',
                size=15,
                zoom=12,
                use_container_width=True
            )
            st.write("**Area Legend**")
            for cluster, color in cluster_colors.items():
                if cluster in map_df['cluster'].unique():
                    cluster_name = clustered_df[clustered_df['cluster'] == cluster]['cluster_name'].iloc[0]
                    st.markdown(f"<span style='color:{color}'>■</span> {cluster_name}", unsafe_allow_html=True)
            st.markdown("<span style='color:#000000'>■</span> Selected Location", unsafe_allow_html=True)
            st.markdown("<span style='color:#0000FF'>■</span> Search Radius", unsafe_allow_html=True)
        else:
            st.warning("No data available to display on map.")
    with tab2:
        show_predictive_modeling(df)
    with tab3:
        show_budget_analysis(df, selected_area)

# Check input change
def check_input_change(input_method, area_name, area_input, lat, lng, radius, selected_cuisines):
    current_inputs = {
        'input_method': input_method,
        'area_name': area_name,
        'area_input': area_input,
        'lat': lat,
        'lng': lng,
        'radius': radius,
        'selected_cuisines': tuple(selected_cuisines)
    }
    if st.session_state.last_inputs != current_inputs:
        st.session_state.show_output = False
        st.session_state.restaurants = pd.DataFrame()
        st.session_state.suggestion = ""
        st.session_state.clustered = False
        st.session_state.last_inputs = current_inputs

# Main function
def main():
    st.title("🍽️ Restaurant Planner")
    st.write("A comprehensive tool for analyzing restaurant locations, competition, and market potential.")
    st.write("### Step 1: Choose Your Location")
    input_method = st.radio("Select input method:", 
                           ("Select Predefined Area", "Enter Area Name", "Enter Latitude/Longitude"),
                           horizontal=True,
                           key="input_method")
    area_name = ""
    area_input = ""
    st.session_state.selected_cuisines=None
    lat = st.session_state.marker_location[0]
    lng = st.session_state.marker_location[1]
    if input_method == "Select Predefined Area":
        area_name = st.selectbox("Select an area", list(AREA_COORDINATES.keys()), index=0, key="area_select")
        st.session_state.selected_area = area_name
        lat, lng = AREA_COORDINATES[area_name]["lat"], AREA_COORDINATES[area_name]["lng"]
        st.session_state.marker_location = [lat, lng]
        st.session_state.selected_table = AREA_COORDINATES[area_name]["table"]
        if st.button("🔍 Fetch Data"):
            with st.spinner("Fetching cuisines..."):
                df, cuisines = load_data(st.session_state.selected_table)
                st.session_state.all_cuisines = cuisines
                if cuisines:
                    st.success(f"Fetched data and loaded cuisines for {area_name}")
                else:
                    st.warning("No cuisines found for this area")
    elif input_method == "Enter Area Name":
        area_input = st.text_input("Enter area name (e.g., Koradi, Nagpur)", key="area_input")
        st.session_state.selected_area = area_input if area_input else "Custom Location"
        if area_input in AREA_COORDINATES:
            st.session_state.selected_area = area_input
            st.session_state.marker_location = [AREA_COORDINATES[area_input]["lat"], AREA_COORDINATES[area_input]["lng"]]
            st.session_state.selected_table = AREA_COORDINATES[area_input]["table"]
            if st.button("🔍 Fetch Data"):
                with st.spinner("Fetching cuisines..."):
                    df, cuisines = load_data(st.session_state.selected_table)
                    st.session_state.all_cuisines = cuisines
                    if cuisines:
                        st.success(f"Fetched data and loaded cuisines for {area_input}")
                    else:
                        st.warning(f"No cuisines found for {area_input}")
        else:
            if st.button("🔍 Fetch Data"):
                st.error("Area not found in predefined locations and API not working: Unable to fetch data.")
            st.session_state.marker_location = [default_latitude, default_longitude]
            st.session_state.selected_table = "CLEANED_NANDANVAN"
            st.session_state.all_cuisines = []
    else:
        col1, col2 = st.columns([2, 2])
        with col1:
            lat = st.number_input("Latitude", value=st.session_state.marker_location[0], format="%.6f", key="lat_input")
        with col2:
            lng = st.number_input("Longitude", value=st.session_state.marker_location[1], format="%.6f", key="lng_input")
        matched_area, matched_coords = find_closest_area(lat, lng)
        if matched_area:
            st.session_state.selected_area = matched_area
            st.session_state.marker_location = [matched_coords["lat"], matched_coords["lng"]]
            st.session_state.selected_table = matched_coords["table"]
            if st.button("🔍 Fetch Data"):
                with st.spinner("Fetching cuisines..."):
                    df, cuisines = load_data(st.session_state.selected_table)
                    st.session_state.all_cuisines = cuisines
                    if cuisines:
                        st.success(f"Fetched data and loaded cuisines for {matched_area}")
                    else:
                        st.warning(f"No cuisines found for {matched_area}")
        else:
            if st.button("🔍 Fetch Data"):
                st.error("Coordinates not found in predefined locations and API not working: Unable to fetch area data.")
            st.session_state.marker_location = [lat, lng]
            st.session_state.selected_area = "Custom Location"
            st.session_state.selected_table = "CLEANED_NANDANVAN"
            st.session_state.all_cuisines = []
    st.write("### Step 2: Set Filters")
    col1, col2 = st.columns(2)
    with col1:
        radius = st.number_input("Radius (KM)", value=st.session_state.radius, min_value=1, max_value=20, key="radius_input")
        st.session_state.radius = radius
    with col2:
        selected_cuisines = st.multiselect(
            "Select cuisines to analyze",
            options=st.session_state.all_cuisines,
            default=st.session_state.selected_cuisines,
            help="Select one or more cuisines to focus your analysis",
            key="cuisine_select"
        )
        st.session_state.selected_cuisines = selected_cuisines
    check_input_change(input_method, area_name, area_input, lat, lng, radius, selected_cuisines)
    st.write("**Quick Tip**: Click 'Get Suggestions' for a quick recommendation or 'Analyze Area' for a detailed analysis.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Analyze Area", type="primary"):
            with st.spinner("Fetching and analyzing restaurant data..."):
                df = fetch_places_data(
                    st.session_state.marker_location[0],
                    st.session_state.marker_location[1],
                    radius,
                    st.session_state.selected_table,
                    st.session_state.selected_cuisines if st.session_state.selected_cuisines else None
                )
                st.session_state.restaurants = df
                st.session_state.clustered = False
                st.session_state.show_output = True
                st.session_state.suggestion = ""
                st.success(f"Found {len(df)} restaurants in the area!")
    with col2:
        if st.button(
            "💡 Get Suggestions",
            type="primary",
            help="Discover the best location and cuisine based on market analysis."
        ):
            with st.spinner("Generating suggestion..."):
                df = fetch_places_data(
                    st.session_state.marker_location[0],
                    st.session_state.marker_location[1],
                    radius,
                    st.session_state.selected_table,
                    st.session_state.selected_cuisines if st.session_state.selected_cuisines else None
                )
                st.session_state.suggestion = get_quick_suggestion(df, st.session_state.selected_cuisines)
                st.session_state.restaurants = df
                st.session_state.show_output = True
                st.session_state.clustered = False
    if st.session_state.show_output:
        if st.session_state.suggestion:
            suggestion_text, recommended_cluster, clustered_df, centroid_lat, centroid_lon, cluster_radius = st.session_state.suggestion
            st.markdown(suggestion_text, unsafe_allow_html=True)
            if recommended_cluster and not clustered_df.empty and centroid_lat is not None:
                st.write("### 🗺️ Recommended Area")
                map_df = clustered_df[clustered_df['cluster_name'] == recommended_cluster][['latitude', 'longitude', 'cluster', 'cluster_name']].copy()
                map_df = map_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
                map_df['color'] = "#FF0000"
                centroid_df = pd.DataFrame({
                    'lat': [centroid_lat],
                    'lon': [centroid_lon],
                    'cluster': [-1],
                    'cluster_name': ['Recommended Area Center'],
                    'color': ["#FFFFFF"]
                })
                combined_df = pd.concat([map_df, centroid_df], ignore_index=True)
                zoom_level = max(12 - int((cluster_radius or 1) / 2), 10)
                st.map(
                    combined_df,
                    latitude='lat',
                    longitude='lon',
                    color='color',
                    size=15,
                    zoom=zoom_level,
                    use_container_width=True
                )
                st.write("**Map Legend**")
                st.markdown("<span style='color:#FF0000'>■</span> Restaurants in Recommended Cluster", unsafe_allow_html=True)
                st.markdown("<span style='color:#FFFFFF'>■</span> Recommended Area Center 📍", unsafe_allow_html=True)
        if not st.session_state.restaurants.empty and not st.session_state.suggestion:
            show_analysis_dashboard(
                st.session_state.restaurants,
                st.session_state.selected_area,
                st.session_state.radius,
                st.session_state.selected_cuisines
            )
    if st.button("🔄 Reset Analysis"):
        st.session_state.restaurants = pd.DataFrame()
        st.session_state.marker_location = [default_latitude, default_longitude]
        st.session_state.selected_area = "Nandanvan, Nagpur"
        st.session_state.clustered = False
        st.session_state.selected_cuisines = []
        st.session_state.all_cuisines = []
        st.session_state.selected_table = "CLEANED_NANDANVAN"
        st.session_state.radius = 5
        st.session_state.show_output = False
        st.session_state.suggestion = ""
        st.session_state.last_inputs = {}
        st.rerun()

if __name__ == "__main__":
    main()
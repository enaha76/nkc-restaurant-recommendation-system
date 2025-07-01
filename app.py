import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="🍽️ Mauritania Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ecdc4;
        margin: 0.5rem 0;
    }
    .restaurant-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MauritaniaRestaurantRecommender:
    def __init__(self, model_package):
        self.model_package = model_package
        self.ridge_model = model_package['ridge_model']
        self.scaler = model_package['scaler']
        self.svd_model = model_package['svd_model']
        self.svd_reconstructed = model_package['svd_reconstructed']
        self.global_avg = model_package['global_avg']
        self.feature_columns = model_package['feature_columns']
        self.train_matrix = model_package['train_matrix']
        self.user_review_counts = model_package['user_review_counts']
        self.restaurant_review_counts = model_package['restaurant_review_counts']
        self.restaurant_features = model_package['restaurant_features']
        self.reviews_clean = model_package['reviews_clean']

    def calculate_hybrid_weight(self, user_id, restaurant_id):
        user_reviews = self.user_review_counts.get(user_id, 0)
        restaurant_reviews = self.restaurant_review_counts.get(restaurant_id, 0)
        
        content_weight = 0.3
        if user_reviews == 0: 
            content_weight += 0.4
        elif user_reviews < 3: 
            content_weight += 0.2
        if restaurant_reviews == 0: 
            content_weight += 0.3
        elif restaurant_reviews < 5: 
            content_weight += 0.1
        
        return min(1.0, content_weight)

    def get_svd_prediction(self, user_id, restaurant_id):
        if user_id in self.train_matrix.index and restaurant_id in self.train_matrix.columns:
            user_idx = self.train_matrix.index.get_loc(user_id)
            restaurant_idx = self.train_matrix.columns.get_loc(restaurant_id)
            return self.svd_reconstructed[user_idx, restaurant_idx]
        return self.global_avg

    def prepare_features_for_prediction(self, user_id, restaurant_id):
        sample_review = self.reviews_clean[
            (self.reviews_clean['reviewer_id'] == user_id) & 
            (self.reviews_clean['restaurant_id'] == restaurant_id)
        ]
        
        if len(sample_review) > 0:
            features = sample_review[self.feature_columns].iloc[0].values
        else:
            features = np.zeros(len(self.feature_columns))
            if restaurant_id in self.restaurant_features.index:
                restaurant_info = self.restaurant_features.loc[restaurant_id]
                if 'latitude' in self.feature_columns:
                    features[self.feature_columns.index('latitude')] = restaurant_info.get('latitude', 18.1)
                if 'longitude' in self.feature_columns:
                    features[self.feature_columns.index('longitude')] = restaurant_info.get('longitude', -15.95)
                if 'total_score' in self.feature_columns:
                    features[self.feature_columns.index('total_score')] = restaurant_info.get('total_score', self.global_avg)
        
        return features.reshape(1, -1)

    def predict_rating(self, user_id, restaurant_id):
        content_weight = self.calculate_hybrid_weight(user_id, restaurant_id)
        svd_pred = self.get_svd_prediction(user_id, restaurant_id)
        
        features = self.prepare_features_for_prediction(user_id, restaurant_id)
        features_scaled = self.scaler.transform(features)
        ridge_pred = self.ridge_model.predict(features_scaled)[0]
        
        hybrid_pred = (content_weight * ridge_pred) + ((1 - content_weight) * svd_pred)
        hybrid_pred = np.clip(hybrid_pred, 1, 5)
        
        return {
            'predicted_rating': round(hybrid_pred, 2),
            'content_weight': round(content_weight, 3),
            'svd_prediction': round(svd_pred, 2),
            'content_prediction': round(ridge_pred, 2),
            'confidence': 'High' if content_weight > 0.6 else 'Medium' if content_weight > 0.4 else 'Low'
        }

    def recommend_restaurants(self, user_id, top_k=5, city_filter=None):
        predictions = []
        
        user_reviews = self.reviews_clean[self.reviews_clean['reviewer_id'] == user_id]
        rated_restaurants = set(user_reviews['restaurant_id'].tolist()) if len(user_reviews) > 0 else set()
        
        for restaurant_id in self.restaurant_features.index:
            if restaurant_id not in rated_restaurants:
                try:
                    restaurant_info = self.restaurant_features.loc[restaurant_id]
                    
                    if city_filter and restaurant_info['city'] != city_filter:
                        continue
                        
                    prediction = self.predict_rating(user_id, restaurant_id)
                    
                    predictions.append({
                        'restaurant_id': restaurant_id,
                        'restaurant_name': restaurant_info['title'],
                        'city': restaurant_info['city'],
                        'predicted_rating': prediction['predicted_rating'],
                        'actual_avg_rating': round(restaurant_info['avg_rating'], 2),
                        'review_count': int(restaurant_info['review_count']),
                        'confidence': prediction['confidence'],
                        'neighborhood': int(restaurant_info['neighborhood']),
                        'latitude': restaurant_info['latitude'],
                        'longitude': restaurant_info['longitude']
                    })
                except Exception as e:
                    continue
        
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return predictions[:top_k]

    def get_restaurant_info(self, restaurant_id):
        if restaurant_id in self.restaurant_features.index:
            info = self.restaurant_features.loc[restaurant_id]
            return {
                'name': info['title'],
                'city': info['city'],
                'avg_rating': round(info['avg_rating'], 2),
                'review_count': int(info['review_count']),
                'neighborhood': int(info['neighborhood']),
                'latitude': info['latitude'],
                'longitude': info['longitude']
            }
        return None

    def get_user_stats(self, user_id):
        user_reviews = self.reviews_clean[self.reviews_clean['reviewer_id'] == user_id]
        if len(user_reviews) > 0:
            return {
                'total_reviews': len(user_reviews),
                'avg_rating_given': round(user_reviews['stars'].mean(), 2),
                'restaurants_reviewed': user_reviews['restaurant_id'].nunique(),
                'cities_visited': user_reviews['city'].nunique(),
                'has_text_reviews': user_reviews['has_text'].sum(),
                'is_active_reviewer': len(user_reviews) >= 3,
                'favorite_city': user_reviews['city'].mode().iloc[0] if len(user_reviews) > 0 else 'Unknown'
            }
        return {'total_reviews': 0, 'is_new_user': True}

    def get_all_users(self):
        return sorted(list(self.reviews_clean['reviewer_id'].unique()))
    
    def get_all_restaurants(self):
        return sorted(list(self.restaurant_features.index))
    
    def get_cities(self):
        return sorted(list(self.restaurant_features['city'].unique()))

# Load model function with robust path handling
@st.cache_resource
def load_recommender_model():
    """Load the trained recommendation model with multiple fallback options"""
    
    # Try multiple possible paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        'mauritania_restaurant_recommender.pkl',
        './mauritania_restaurant_recommender.pkl',
        os.path.join(script_dir, 'mauritania_restaurant_recommender.pkl'),
        os.path.join(os.getcwd(), 'mauritania_restaurant_recommender.pkl')
    ]
    
    # Debug information
    st.sidebar.write("🔍 **Debug Info:**")
    st.sidebar.write(f"Script directory: `{script_dir}`")
    st.sidebar.write(f"Working directory: `{os.getcwd()}`")
    st.sidebar.write(f"Files in directory: {os.listdir('.')}")
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    model_package = pickle.load(f)
                st.sidebar.success(f"✅ Model loaded from: `{path}`")
                return MauritaniaRestaurantRecommender(model_package)
            except Exception as e:
                st.sidebar.error(f"❌ Error loading from `{path}`: {e}")
                continue
    
    # If no file found, show file uploader
    st.sidebar.error("❌ Model file not found in any expected location.")
    st.sidebar.info("📁 Please upload your model file below:")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Model File",
        type=['pkl'],
        help="Upload the mauritania_restaurant_recommender.pkl file",
        key="model_uploader"
    )
    
    if uploaded_file is not None:
        try:
            model_package = pickle.load(uploaded_file)
            st.sidebar.success("✅ Model loaded from uploaded file!")
            return MauritaniaRestaurantRecommender(model_package)
        except Exception as e:
            st.sidebar.error(f"❌ Error loading uploaded model: {e}")
            return None
    
    return None

def display_restaurant_card(rec, index):
    """Display a restaurant recommendation card"""
    confidence_color = {
        'High': '#28a745',
        'Medium': '#ffc107', 
        'Low': '#dc3545'
    }
    
    st.markdown(f"""
    <div class="restaurant-card">
        <h4>{index}. 🍽️ {rec['restaurant_name']}</h4>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p><strong>📍 Location:</strong> {rec['city']} (Neighborhood {rec['neighborhood']})</p>
                <p><strong>🔮 Predicted Rating:</strong> ⭐ {rec['predicted_rating']}/5.0</p>
                <p><strong>📊 Actual Average:</strong> ⭐ {rec['actual_avg_rating']}/5.0 ({rec['review_count']} reviews)</p>
            </div>
            <div style="text-align: right;">
                <span style="background: {confidence_color.get(rec['confidence'], '#6c757d')}; color: white; padding: 0.2rem 0.5rem; border-radius: 5px; font-size: 0.8rem;">
                    {rec['confidence']} Confidence
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_recommendations_map(recommendations):
    """Create an interactive map of restaurant recommendations"""
    if not recommendations:
        return None
    
    map_data = pd.DataFrame([{
        'name': rec['restaurant_name'][:30] + ('...' if len(rec['restaurant_name']) > 30 else ''),
        'lat': rec['latitude'],
        'lon': rec['longitude'],
        'predicted_rating': rec['predicted_rating'],
        'city': rec['city'],
        'full_name': rec['restaurant_name']
    } for rec in recommendations])
    
    fig = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        hover_name="full_name",
        hover_data=["predicted_rating", "city"],
        color="predicted_rating",
        size="predicted_rating",
        color_continuous_scale="Viridis",
        size_max=15,
        zoom=10,
        height=400,
        title="🗺️ Recommended Restaurants Map"
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ Mauritania Restaurant Recommender</h1>
        <p>AI-Powered Personalized Restaurant Recommendations for Mauritania</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    recommender = load_recommender_model()
    
    if recommender is None:
        st.error("🚫 Cannot proceed without a trained model. Please upload your model file in the sidebar.")
        st.info("💡 **How to get the model file:**\n1. Run your training notebook\n2. Save the model using the provided code\n3. Upload the `.pkl` file using the sidebar uploader")
        st.stop()

    # Model Info Sidebar
    with st.sidebar:
        st.markdown("### 🤖 Model Information")
        try:
            performance_info = f"""
            <div class="metric-card">
                <strong>Performance:</strong><br>
                RMSE: {recommender.model_package['final_rmse']:.3f}<br>
                Improvement: +{recommender.model_package['improvement_vs_baseline']:.1f}%
            </div>
            """
            st.markdown(performance_info, unsafe_allow_html=True)
        except:
            st.info("Model performance metrics not available")
        
        st.markdown("### 📊 Dataset Stats")
        try:
            total_restaurants = len(recommender.get_all_restaurants())
            total_users = len(recommender.get_all_users())
            total_reviews = len(recommender.reviews_clean)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Restaurants", total_restaurants)
                st.metric("Users", total_users)
            with col2:
                st.metric("Reviews", total_reviews)
                st.metric("Cities", len(recommender.get_cities()))
        except Exception as e:
            st.error(f"Error loading stats: {e}")

    # Main Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Get Recommendations", 
        "🔮 Predict Rating", 
        "🏪 Explore Restaurants",
        "👤 User Analysis",
        "📈 Analytics Dashboard"
    ])

    with tab1:
        st.header("🎯 Personalized Restaurant Recommendations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # User selection
            try:
                all_users = recommender.get_all_users()
                selected_user = st.selectbox(
                    "Select a User ID:",
                    options=all_users,
                    help="Choose a user to get personalized recommendations"
                )
                
                # Filters
                col_a, col_b = st.columns(2)
                with col_a:
                    cities = recommender.get_cities()
                    city_filter = st.selectbox(
                        "Filter by City (Optional):",
                        options=['All Cities'] + cities
                    )
                with col_b:
                    num_recommendations = st.slider("Number of Recommendations:", 1, 10, 5)
            except Exception as e:
                st.error(f"Error setting up filters: {e}")
                return
        
        with col2:
            # User stats
            if selected_user:
                try:
                    user_stats = recommender.get_user_stats(selected_user)
                    st.markdown("### 👤 User Profile")
                    if user_stats.get('total_reviews', 0) > 0:
                        st.write(f"**Reviews:** {user_stats['total_reviews']}")
                        st.write(f"**Avg Rating:** ⭐ {user_stats['avg_rating_given']}")
                        st.write(f"**Restaurants:** {user_stats['restaurants_reviewed']}")
                        st.write(f"**Favorite City:** {user_stats['favorite_city']}")
                    else:
                        st.write("**New User** 🆕")
                        st.write("Perfect for content-based recommendations!")
                except Exception as e:
                    st.error(f"Error loading user stats: {e}")

        # Get recommendations
        if st.button("🚀 Get Recommendations", type="primary"):
            city_param = None if city_filter == 'All Cities' else city_filter
            
            try:
                with st.spinner("🔄 Generating personalized recommendations..."):
                    recommendations = recommender.recommend_restaurants(
                        selected_user, 
                        top_k=num_recommendations,
                        city_filter=city_param
                    )
                
                if recommendations:
                    st.success(f"✅ Found {len(recommendations)} recommendations!")
                    
                    # Display recommendations
                    for i, rec in enumerate(recommendations, 1):
                        display_restaurant_card(rec, i)
                    
                    # Map visualization
                    st.subheader("🗺️ Recommendations Map")
                    map_fig = create_recommendations_map(recommendations)
                    if map_fig:
                        st.plotly_chart(map_fig, use_container_width=True)
                else:
                    st.warning("⚠️ No recommendations found with current filters.")
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

    with tab2:
        st.header("🔮 Rating Prediction")
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                all_users = recommender.get_all_users()
                pred_user = st.selectbox(
                    "Select User:",
                    options=all_users,
                    key="pred_user"
                )
            
            with col2:
                all_restaurants = recommender.get_all_restaurants()
                pred_restaurant = st.selectbox(
                    "Select Restaurant:",
                    options=all_restaurants,
                    format_func=lambda x: f"{x}: {recommender.get_restaurant_info(x)['name'][:30]}..." if recommender.get_restaurant_info(x) else str(x),
                    key="pred_restaurant"
                )
            
            if st.button("🎯 Predict Rating", type="primary"):
                try:
                    with st.spinner("🔄 Calculating prediction..."):
                        prediction = recommender.predict_rating(pred_user, pred_restaurant)
                        restaurant_info = recommender.get_restaurant_info(pred_restaurant)
                    
                    # Display prediction in a nice box
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🔮 Prediction Result</h2>
                        <h1>⭐ {prediction['predicted_rating']}/5.0</h1>
                        <p><strong>Confidence:</strong> {prediction['confidence']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed breakdown
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Content-Based Prediction",
                            f"⭐ {prediction['content_prediction']}/5.0",
                            help="Rating based on restaurant and user features"
                        )
                    
                    with col2:
                        st.metric(
                            "Collaborative Prediction",
                            f"⭐ {prediction['svd_prediction']}/5.0",
                            help="Rating based on similar users' preferences"
                        )
                    
                    with col3:
                        st.metric(
                            "Content Weight",
                            f"{prediction['content_weight']*100:.1f}%",
                            help="How much the model relies on content vs collaborative filtering"
                        )
                    
                    # Restaurant details
                    if restaurant_info:
                        st.subheader("🏪 Restaurant Details")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Avg Rating", f"⭐ {restaurant_info['avg_rating']}/5.0")
                        with col2:
                            st.metric("Total Reviews", restaurant_info['review_count'])
                        with col3:
                            st.metric("City", restaurant_info['city'])
                        with col4:
                            st.metric("Neighborhood", restaurant_info['neighborhood'])
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        except Exception as e:
            st.error(f"Error setting up prediction interface: {e}")

    with tab3:
        st.header("🏪 Restaurant Explorer")
        
        try:
            # Restaurant search and filter
            col1, col2 = st.columns([2, 1])
            
            with col1:
                cities = recommender.get_cities()
                search_city = st.selectbox(
                    "Filter by City:",
                    options=['All Cities'] + cities,
                    key="explore_city"
                )
            
            with col2:
                sort_by = st.selectbox(
                    "Sort by:",
                    options=['Average Rating', 'Review Count', 'Neighborhood'],
                    key="sort_restaurants"
                )
            
            # Get restaurants
            all_restaurants = recommender.restaurant_features.copy()
            
            if search_city != 'All Cities':
                all_restaurants = all_restaurants[all_restaurants['city'] == search_city]
            
            # Sort restaurants
            sort_mapping = {
                'Average Rating': 'avg_rating',
                'Review Count': 'review_count',
                'Neighborhood': 'neighborhood'
            }
            all_restaurants = all_restaurants.sort_values(sort_mapping[sort_by], ascending=False)
            
            # Display restaurants
            st.subheader(f"🍽️ Restaurants ({len(all_restaurants)} found)")
            
            for idx, (restaurant_id, restaurant) in enumerate(all_restaurants.head(20).iterrows()):
                with st.expander(f"🏪 {restaurant['title']} - ⭐ {restaurant['avg_rating']:.2f}/5.0"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**📍 City:** {restaurant['city']}")
                        st.write(f"**🏘️ Neighborhood:** {restaurant['neighborhood']}")
                    
                    with col2:
                        st.write(f"**⭐ Average Rating:** {restaurant['avg_rating']:.2f}/5.0")
                        st.write(f"**📊 Total Reviews:** {restaurant['review_count']}")
                    
                    with col3:
                        st.write(f"**🌐 Latitude:** {restaurant['latitude']:.4f}")
                        st.write(f"**🌐 Longitude:** {restaurant['longitude']:.4f}")
        except Exception as e:
            st.error(f"Error in restaurant explorer: {e}")

    with tab4:
        st.header("👤 User Analysis")
        
        try:
            all_users = recommender.get_all_users()
            analysis_user = st.selectbox(
                "Select User for Analysis:",
                options=all_users,
                key="analysis_user"
            )
            
            if st.button("📊 Analyze User", type="primary"):
                try:
                    user_stats = recommender.get_user_stats(analysis_user)
                    user_reviews = recommender.reviews_clean[recommender.reviews_clean['reviewer_id'] == analysis_user]
                    
                    if user_stats.get('total_reviews', 0) > 0:
                        # User metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Reviews", user_stats['total_reviews'])
                        
                        with col2:
                            st.metric("Average Rating Given", f"⭐ {user_stats['avg_rating_given']}/5.0")
                        
                        with col3:
                            st.metric("Restaurants Reviewed", user_stats['restaurants_reviewed'])
                        
                        with col4:
                            st.metric("Cities Visited", user_stats['cities_visited'])
                        
                        # Rating distribution
                        st.subheader("📊 User's Rating Distribution")
                        rating_dist = user_reviews['stars'].value_counts().sort_index()
                        
                        fig = px.bar(
                            x=rating_dist.index,
                            y=rating_dist.values,
                            labels={'x': 'Rating', 'y': 'Number of Reviews'},
                            title="Distribution of Ratings Given by User"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Recent reviews
                        st.subheader("📝 Recent Reviews")
                        recent_reviews = user_reviews.head(5)
                        
                        for _, review in recent_reviews.iterrows():
                            restaurant_info = recommender.get_restaurant_info(review['restaurant_id'])
                            if restaurant_info:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <strong>🏪 {restaurant_info['name']}</strong><br>
                                    ⭐ Rating: {review['stars']}/5.0<br>
                                    📍 City: {restaurant_info['city']}<br>
                                    📝 Has Text: {'Yes' if review['has_text'] else 'No'}
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("👤 This is a new user with no review history.")
                        st.write("Perfect candidate for content-based recommendations!")
                except Exception as e:
                    st.error(f"Error analyzing user: {e}")
        except Exception as e:
            st.error(f"Error setting up user analysis: {e}")

    with tab5:
        st.header("📈 Analytics Dashboard")
        
        try:
            # Overall statistics
            st.subheader("🌍 Overall Platform Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Restaurants",
                    len(recommender.restaurant_features),
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Total Users",
                    len(recommender.get_all_users()),
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Total Reviews",
                    len(recommender.reviews_clean),
                    delta=None
                )
            
            with col4:
                avg_rating = recommender.reviews_clean['stars'].mean()
                st.metric(
                    "Average Rating",
                    f"⭐ {avg_rating:.2f}/5.0",
                    delta=None
                )
            
            # City distribution
            st.subheader("🏙️ Restaurant Distribution by City")
            city_counts = recommender.restaurant_features['city'].value_counts()
            
            fig = px.pie(
                values=city_counts.values,
                names=city_counts.index,
                title="Restaurants by City"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Rating distribution
            st.subheader("⭐ Rating Distribution")
            rating_dist = recommender.reviews_clean['stars'].value_counts().sort_index()
            
            fig = px.bar(
                x=[f"⭐ {i}" for i in rating_dist.index],
                y=rating_dist.values,
                labels={'x': 'Rating', 'y': 'Number of Reviews'},
                title="Distribution of All Ratings"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Neighborhood analysis
            st.subheader("🏘️ Neighborhood Analysis")
            neighborhood_stats = recommender.restaurant_features.groupby('neighborhood').agg({
                'avg_rating': 'mean',
                'review_count': 'sum',
                'title': 'count'
            }).round(2)
            neighborhood_stats.columns = ['Avg Rating', 'Total Reviews', 'Restaurant Count']
            
            st.dataframe(neighborhood_stats, use_container_width=True)
        except Exception as e:
            st.error(f"Error in analytics dashboard: {e}")

if __name__ == "__main__":
    main()

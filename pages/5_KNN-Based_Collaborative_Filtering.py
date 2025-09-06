# pages/5_KNN-Based_Collaborative_Filtering.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="KNN Collaborative Filtering", layout="wide")
st.title("🤝 KNN-Based Collaborative Filtering")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    courses = pd.read_csv("data/course_genre.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return courses, ratings

courses, ratings = load_data()

# Create user-item matrix
@st.cache_data
def create_user_item_matrix(ratings_df):
    """Create user-item rating matrix"""
    user_item_matrix = ratings_df.pivot_table(
        index='user', 
        columns='item', 
        values='rating',
        fill_value=0
    )
    return user_item_matrix

# Create sparse matrix for efficiency
@st.cache_data
def create_sparse_matrix(user_item_matrix):
    """Convert to sparse matrix for efficient computation"""
    return csr_matrix(user_item_matrix.values)

# Train KNN models
@st.cache_data
def train_knn_models(_user_item_sparse, _item_user_sparse, n_neighbors=10):
    """Train both user-based and item-based KNN models"""
    
    # User-based KNN
    user_knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1
    )
    user_knn.fit(_user_item_sparse)
    
    # Item-based KNN
    item_knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1
    )
    item_knn.fit(_item_user_sparse)
    
    return user_knn, item_knn

# User-based recommendations
def get_user_based_recommendations(user_id, user_item_matrix, user_knn, n_neighbors=5, n_recommendations=10):
    """Generate recommendations using user-based collaborative filtering"""
    
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(), [], "User not found"
    
    # Get user index
    user_idx = user_item_matrix.index.get_loc(user_id)
    
    # Find similar users
    distances, indices = user_knn.kneighbors(
        user_item_matrix.iloc[user_idx].values.reshape(1, -1),
        n_neighbors=n_neighbors + 1
    )
    
    # Remove the user itself from neighbors
    similar_user_indices = indices.flatten()[1:]
    similar_user_distances = distances.flatten()[1:]
    similar_user_ids = user_item_matrix.index[similar_user_indices].tolist()
    
    # Get courses rated by user
    user_rated_courses = user_item_matrix.iloc[user_idx][user_item_matrix.iloc[user_idx] > 0].index.tolist()
    
    # Aggregate ratings from similar users
    recommendations = {}
    for idx, similar_user_idx in enumerate(similar_user_indices):
        similarity = 1 - similar_user_distances[idx]  # Convert distance to similarity
        similar_user_ratings = user_item_matrix.iloc[similar_user_idx]
        
        for course_id in similar_user_ratings[similar_user_ratings > 0].index:
            if course_id not in user_rated_courses:
                if course_id not in recommendations:
                    recommendations[course_id] = {'weighted_rating': 0, 'similarity_sum': 0, 'count': 0}
                
                recommendations[course_id]['weighted_rating'] += similar_user_ratings[course_id] * similarity
                recommendations[course_id]['similarity_sum'] += similarity
                recommendations[course_id]['count'] += 1
    
    # Calculate predicted ratings
    for course_id in recommendations:
        if recommendations[course_id]['similarity_sum'] > 0:
            recommendations[course_id]['predicted_rating'] = (
                recommendations[course_id]['weighted_rating'] / 
                recommendations[course_id]['similarity_sum']
            )
        else:
            recommendations[course_id]['predicted_rating'] = 0
    
    # Create recommendations dataframe
    recs_df = pd.DataFrame.from_dict(recommendations, orient='index')
    if not recs_df.empty:
        recs_df = recs_df.sort_values('predicted_rating', ascending=False).head(n_recommendations)
        recs_df['course_id'] = recs_df.index
        
        # Add course titles
        recs_df = recs_df.merge(courses[['COURSE_ID', 'TITLE']], left_on='course_id', right_on='COURSE_ID')
        
        # Add actual average ratings
        course_ratings = ratings.groupby('item')['rating'].agg(['mean', 'count']).reset_index()
        recs_df = recs_df.merge(course_ratings, left_on='course_id', right_on='item', how='left')
        
        return recs_df, similar_user_ids, None
    
    return pd.DataFrame(), similar_user_ids, "No recommendations found"

# Item-based recommendations
def get_item_based_recommendations(user_id, user_item_matrix, item_knn, n_recommendations=10):
    """Generate recommendations using item-based collaborative filtering"""
    
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(), "User not found"
    
    # Get user's rated items
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0]
    
    if len(rated_items) == 0:
        return pd.DataFrame(), "User has no ratings"
    
    # Find similar items for each rated item
    recommendations = {}
    
    for item_id, rating in rated_items.items():
        item_idx = user_item_matrix.columns.get_loc(item_id)
        
        # Find similar items
        distances, indices = item_knn.kneighbors(
            user_item_matrix.T.iloc[item_idx].values.reshape(1, -1),
            n_neighbors=11  # Get extra to filter out the item itself
        )
        
        similar_items = []
        for idx, dist in zip(indices.flatten()[1:], distances.flatten()[1:]):
            similar_item_id = user_item_matrix.columns[idx]
            if similar_item_id not in rated_items.index:
                similarity = 1 - dist
                if similar_item_id not in recommendations:
                    recommendations[similar_item_id] = {'weighted_score': 0, 'weight_sum': 0}
                
                recommendations[similar_item_id]['weighted_score'] += rating * similarity
                recommendations[similar_item_id]['weight_sum'] += similarity
    
    # Calculate predicted ratings
    for item_id in recommendations:
        if recommendations[item_id]['weight_sum'] > 0:
            recommendations[item_id]['predicted_rating'] = (
                recommendations[item_id]['weighted_score'] / 
                recommendations[item_id]['weight_sum']
            )
    
    # Create recommendations dataframe
    recs_df = pd.DataFrame.from_dict(recommendations, orient='index')
    if not recs_df.empty:
        recs_df = recs_df.sort_values('predicted_rating', ascending=False).head(n_recommendations)
        recs_df['course_id'] = recs_df.index
        
        # Add course titles
        recs_df = recs_df.merge(courses[['COURSE_ID', 'TITLE']], left_on='course_id', right_on='COURSE_ID')
        
        # Add actual ratings
        course_ratings = ratings.groupby('item')['rating'].agg(['mean', 'count']).reset_index()
        recs_df = recs_df.merge(course_ratings, left_on='course_id', right_on='item', how='left')
        
        return recs_df, None
    
    return pd.DataFrame(), "No recommendations found"

# Streamlit UI
st.markdown("### How it works")
st.info("""
KNN Collaborative Filtering finds users or items most similar to the target and uses their ratings to predict preferences.
- **User-Based**: Find similar users and recommend what they liked
- **Item-Based**: Find similar items to what the user liked
""")

# Create matrices
with st.spinner("Creating user-item matrix..."):
    user_item_matrix = create_user_item_matrix(ratings)
    user_item_sparse = create_sparse_matrix(user_item_matrix)
    item_user_sparse = csr_matrix(user_item_matrix.T.values)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    approach = st.radio(
        "Filtering Approach",
        ["User-Based", "Item-Based", "Hybrid (Both)"],
        help="Choose the collaborative filtering approach"
    )
    
    n_neighbors = st.slider(
        "Number of Neighbors (K)",
        min_value=3,
        max_value=20,
        value=5,
        help="Number of similar users/items to consider"
    )
    
    n_recommendations = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of recommendations to generate"
    )
    
    st.markdown("---")
    st.subheader("📊 Dataset Statistics")
    st.metric("Total Users", len(user_item_matrix))
    st.metric("Total Items", len(user_item_matrix.columns))
    st.metric("Total Ratings", ratings.shape[0])
    st.metric("Sparsity", f"{(1 - ratings.shape[0] / (len(user_item_matrix) * len(user_item_matrix.columns))):.2%}")

# Train models
with st.spinner("Training KNN models..."):
    user_knn, item_knn = train_knn_models(user_item_sparse, item_user_sparse, n_neighbors)

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Get Recommendations", "👥 Similar Users", "📊 Analysis", "🔍 Evaluation"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Select User")
        
        # User selection with search
        search_user = st.text_input("Search User ID:", "")
        
        if search_user:
            filtered_users = [u for u in user_item_matrix.index.tolist() if str(search_user) in str(u)]
        else:
            filtered_users = user_item_matrix.index.tolist()
        
        if filtered_users:
            user_id = st.selectbox(
                "Select a User ID",
                options=filtered_users[:100],  # Limit for performance
                help="Choose a user to generate recommendations"
            )
            
            # Show user info
            if user_id:
                user_ratings = ratings[ratings['user'] == user_id]
                
                st.markdown("#### 👤 User Statistics")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Total Ratings", len(user_ratings))
                with col_b:
                    st.metric("Avg Rating", f"{user_ratings['rating'].mean():.2f}")
                
                # Show rating distribution
                rating_dist = user_ratings['rating'].value_counts().sort_index()
                fig = px.bar(
                    x=rating_dist.index,
                    y=rating_dist.values,
                    title="User's Rating Distribution",
                    labels={'x': 'Rating', 'y': 'Count'},
                    color=rating_dist.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=False, height=200)
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate recommendations button
                if st.button("🚀 Get Recommendations", type="primary"):
                    with st.spinner("Generating recommendations..."):
                        if approach == "User-Based":
                            recs, similar_users, error = get_user_based_recommendations(
                                user_id, user_item_matrix, user_knn, n_neighbors, n_recommendations
                            )
                            st.session_state['user_recs'] = recs
                            st.session_state['similar_users'] = similar_users
                            st.session_state['item_recs'] = None
                            st.session_state['error'] = error
                            
                        elif approach == "Item-Based":
                            recs, error = get_item_based_recommendations(
                                user_id, user_item_matrix, item_knn, n_recommendations
                            )
                            st.session_state['item_recs'] = recs
                            st.session_state['user_recs'] = None
                            st.session_state['similar_users'] = None
                            st.session_state['error'] = error
                            
                        else:  # Hybrid
                            user_recs, similar_users, user_error = get_user_based_recommendations(
                                user_id, user_item_matrix, user_knn, n_neighbors, n_recommendations
                            )
                            item_recs, item_error = get_item_based_recommendations(
                                user_id, user_item_matrix, item_knn, n_recommendations
                            )
                            st.session_state['user_recs'] = user_recs
                            st.session_state['item_recs'] = item_recs
                            st.session_state['similar_users'] = similar_users
                            st.session_state['error'] = user_error or item_error
                        
                        st.session_state['selected_user'] = user_id
    
    with col2:
        st.subheader("📚 Recommendations")
        
        if 'error' in st.session_state and st.session_state['error']:
            st.error(st.session_state['error'])
        
        # User-based recommendations
        if 'user_recs' in st.session_state and st.session_state['user_recs'] is not None and not st.session_state['user_recs'].empty:
            st.markdown("#### 👥 User-Based Recommendations")
            
            display_df = st.session_state['user_recs'][['TITLE', 'predicted_rating', 'mean', 'count']].copy()
            display_df.columns = ['Course Title', 'Predicted Rating', 'Actual Avg Rating', 'Total Ratings']
            display_df['Predicted Rating'] = display_df['Predicted Rating'].round(2)
            display_df['Actual Avg Rating'] = display_df['Actual Avg Rating'].round(2)
            display_df.index = range(1, len(display_df) + 1)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Visualization
            fig = px.scatter(
                st.session_state['user_recs'],
                x='mean',
                y='predicted_rating',
                size='count',
                hover_data=['TITLE'],
                title='Predicted vs Actual Ratings (User-Based)',
                labels={'mean': 'Actual Average Rating', 'predicted_rating': 'Predicted Rating', 'count': 'Number of Ratings'},
                color='predicted_rating',
                color_continuous_scale='RdYlGn'
            )
            fig.add_trace(go.Scatter(
                x=[1, 5],
                y=[1, 5],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Item-based recommendations
        if 'item_recs' in st.session_state and st.session_state['item_recs'] is not None and not st.session_state['item_recs'].empty:
            st.markdown("#### 📦 Item-Based Recommendations")
            
            display_df = st.session_state['item_recs'][['TITLE', 'predicted_rating', 'mean', 'count']].copy()
            display_df.columns = ['Course Title', 'Predicted Rating', 'Actual Avg Rating', 'Total Ratings']
            display_df['Predicted Rating'] = display_df['Predicted Rating'].round(2)
            display_df['Actual Avg Rating'] = display_df['Actual Avg Rating'].round(2)
            display_df.index = range(1, len(display_df) + 1)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Visualization
            fig = px.scatter(
                st.session_state['item_recs'],
                x='mean',
                y='predicted_rating',
                size='count',
                hover_data=['TITLE'],
                title='Predicted vs Actual Ratings (Item-Based)',
                labels={'mean': 'Actual Average Rating', 'predicted_rating': 'Predicted Rating', 'count': 'Number of Ratings'},
                color='predicted_rating',
                color_continuous_scale='RdYlGn'
            )
            fig.add_trace(go.Scatter(
                x=[1, 5],
                y=[1, 5],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("👥 Similar Users Analysis")
    
    if 'similar_users' in st.session_state and st.session_state['similar_users']:
        st.success(f"Found {len(st.session_state['similar_users'])} similar users")
        
        # Analyze similar users
        similar_users_data = []
        for similar_user in st.session_state['similar_users']:
            user_ratings = ratings[ratings['user'] == similar_user]
            similar_users_data.append({
                'User ID': similar_user,
                'Ratings Count': len(user_ratings),
                'Avg Rating': user_ratings['rating'].mean(),
                'Common Items': len(set(user_ratings['item']) & set(ratings[ratings['user'] == st.session_state.get('selected_user', 0)]['item']))
            })
        
        similar_users_df = pd.DataFrame(similar_users_data)
        similar_users_df = similar_users_df.round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(similar_users_df, use_container_width=True)
        
        with col2:
            # Visualize similarity
            fig = px.bar(
                similar_users_df,
                x='User ID',
                y='Common Items',
                title='Common Items with Similar Users',
                color='Avg Rating',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show what similar users liked
        st.markdown("#### 🎯 What Similar Users Liked")
        similar_user_items = ratings[ratings['user'].isin(st.session_state['similar_users'])]
        top_items = similar_user_items.groupby('item').agg({
            'rating': ['mean', 'count']
        }).round(2)
        top_items.columns = ['avg_rating', 'count']
        top_items = top_items[top_items['count'] >= 2].sort_values('avg_rating', ascending=False).head(10)
        
        if not top_items.empty:
            top_items = top_items.merge(courses[['COURSE_ID', 'TITLE']], left_index=True, right_on='COURSE_ID')
            
            fig = px.bar(
                top_items,
                x='avg_rating',
                y='TITLE',
                orientation='h',
                title='Top Rated Courses by Similar Users',
                color='count',
                color_continuous_scale='Viridis',
                labels={'avg_rating': 'Average Rating', 'TITLE': 'Course', 'count': 'Number of Ratings'}
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Generate user-based recommendations to see similar users")

with tab3:
    st.subheader("📊 Model Analysis")
    
    # Neighbor distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### User Similarity Distribution")
        
        # Sample random users and calculate similarities
        sample_size = min(50, len(user_item_matrix))
        sample_users = np.random.choice(user_item_matrix.index, sample_size, replace=False)
        
        similarities = []
        for user in sample_users:
            user_idx = user_item_matrix.index.get_loc(user)
            distances, _ = user_knn.kneighbors(
                user_item_matrix.iloc[user_idx].values.reshape(1, -1),
                n_neighbors=n_neighbors + 1
            )
            similarities.extend(1 - distances.flatten()[1:])
        
        fig = px.histogram(
            similarities,
            nbins=30,
            title="Distribution of User Similarities",
            labels={'value': 'Similarity Score', 'count': 'Frequency'},
            color_discrete_sequence=['lightblue']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Rating Coverage")
        
        # Calculate coverage
        user_coverage = (user_item_matrix > 0).sum(axis=1)
        item_coverage = (user_item_matrix > 0).sum(axis=0)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Ratings per User', 'Ratings per Item')
        )
        
        fig.add_trace(
            go.Histogram(x=user_coverage, nbinsx=20, marker_color='lightgreen'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=item_coverage, nbinsx=20, marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Number of Ratings", row=1, col=1)
        fig.update_xaxes(title_text="Number of Ratings", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_layout(showlegend=False, height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Sparsity analysis
    st.markdown("#### 🕸️ Sparsity Analysis")
    
    total_possible = len(user_item_matrix) * len(user_item_matrix.columns)
    total_ratings = (user_item_matrix > 0).sum().sum()
    sparsity = 1 - (total_ratings / total_possible)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matrix Sparsity", f"{sparsity:.2%}")
    with col2:
        st.metric("Avg Ratings/User", f"{user_coverage.mean():.1f}")
    with col3:
        st.metric("Avg Ratings/Item", f"{item_coverage.mean():.1f}")
    with col4:
        st.metric("Matrix Density", f"{(1-sparsity):.4%}")

with tab4:
    st.subheader("🔍 Model Evaluation")
    
    st.markdown("#### Cross-Validation Performance")
    
    if st.button("Run Evaluation (This may take a while)"):
        with st.spinner("Evaluating model performance..."):
            # Simple train-test split evaluation
            test_size = 0.2
            test_mask = np.random.random(user_item_matrix.shape) < test_size
            train_data = user_item_matrix.copy()
            train_data[test_mask] = 0
            
            # Re-train on training data
            train_sparse = csr_matrix(train_data.values)
            eval_user_knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
            eval_user_knn.fit(train_sparse)
            
            # Calculate RMSE on test set
            errors = []
            coverage_count = 0
            
            for user in np.random.choice(user_item_matrix.index, min(100, len(user_item_matrix)), replace=False):
                user_idx = user_item_matrix.index.get_loc(user)
                test_items = user_item_matrix.columns[test_mask[user_idx]]
                
                if len(test_items) > 0:
                    # Get predictions
                    distances, indices = eval_user_knn.kneighbors(
                        train_data.iloc[user_idx].values.reshape(1, -1),
                        n_neighbors=min(n_neighbors + 1, len(train_data))
                    )
                    
                    if len(indices.flatten()) > 1:
                        similar_users = indices.flatten()[1:]
                        similar_distances = distances.flatten()[1:]
                        
                        for item in test_items:
                            item_idx = user_item_matrix.columns.get_loc(item)
                            actual = user_item_matrix.iloc[user_idx, item_idx]
                            
                            if actual > 0:
                                # Predict rating
                                predictions = []
                                weights = []
                                
                                for idx, dist in zip(similar_users, similar_distances):
                                    if train_data.iloc[idx, item_idx] > 0:
                                        predictions.append(train_data.iloc[idx, item_idx])
                                        weights.append(1 - dist)
                                
                                if predictions:
                                    predicted = np.average(predictions, weights=weights)
                                    errors.append((actual - predicted) ** 2)
                                    coverage_count += 1
            
            if errors:
                rmse = np.sqrt(np.mean(errors))
                mae = np.mean(np.abs(np.sqrt(errors)))
                coverage = coverage_count / (coverage_count + len(test_mask[test_mask == True]) - coverage_count)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{rmse:.3f}")
                with col2:
                    st.metric("MAE", f"{mae:.3f}")
                with col3:
                    st.metric("Coverage", f"{coverage:.2%}")
                
                # Error distribution
                fig = px.histogram(
                    np.sqrt(errors),
                    nbins=30,
                    title="Prediction Error Distribution",
                    labels={'value': 'Absolute Error', 'count': 'Frequency'},
                    color_discrete_sequence=['lightblue']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data for evaluation")

# Footer
st.markdown("---")
st.markdown("### 💡 Algorithm Details")
with st.expander("Understanding KNN Collaborative Filtering"):
    st.markdown("""
    **How KNN Collaborative Filtering Works:**
    
    **User-Based CF:**
    1. Find K most similar users to the target user
    2. Similarity measured using cosine similarity on rating vectors
    3. Predict ratings as weighted average of similar users' ratings
    4. Recommend items with highest predicted ratings
    
    **Item-Based CF:**
    1. Find items similar to what the user has rated
    2. Similarity based on rating patterns across all users
    3. Predict ratings based on user's ratings of similar items
    4. Recommend items with highest predicted scores
    
    **Key Parameters:**
    - **K (neighbors)**: Number of similar users/items to consider
    - **Similarity Metric**: Cosine similarity (angle between vectors)
    - **Minimum Support**: Minimum ratings needed for recommendations
    
    **Advantages:**
    - Simple and interpretable
    - No training required (lazy learning)
    - Can capture local patterns
    - Works well with sparse data
    
    **Limitations:**
    - Scalability issues with large datasets
    - Sparsity can affect quality
    - Cold start problem for new users/items
    - Memory intensive for large matrices
    """)
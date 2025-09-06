# pages/6_NMF-Based_Collaborative_Filtering.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="NMF Collaborative Filtering", layout="wide")
st.title("🔢 NMF-Based Collaborative Filtering")
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

# Prepare data for NMF
def prepare_nmf_data(user_item_matrix):
    """Prepare and normalize data for NMF"""
    # Replace 0s with small value to maintain non-negativity
    matrix_filled = user_item_matrix.copy()
    matrix_filled[matrix_filled == 0] = 0.1  # Small value for unrated items
    
    # Normalize ratings to [0, 1] range for better NMF performance
    scaler = MinMaxScaler()
    matrix_normalized = pd.DataFrame(
        scaler.fit_transform(matrix_filled),
        index=matrix_filled.index,
        columns=matrix_filled.columns
    )
    
    return matrix_normalized, scaler

# Train NMF model - Simple version without problematic parameters
def train_nmf_simple(matrix_normalized, n_components=20, max_iter=200):
    """Train NMF model and return factors"""
    
    # Create NMF model with only basic parameters
    nmf_model = NMF(
        n_components=n_components,
        random_state=42,
        max_iter=max_iter
    )
    
    # Fit model and get factors
    W = nmf_model.fit_transform(matrix_normalized)  # User features
    H = nmf_model.components_  # Item features
    
    # Calculate reconstruction error manually
    reconstruction = W @ H
    reconstruction_error = np.mean((matrix_normalized.values - reconstruction) ** 2)
    
    return W, H, nmf_model, reconstruction_error

# Get NMF-based recommendations
def get_nmf_recommendations(user_id, user_item_matrix, W, H, scaler, n_recommendations=10):
    """Generate recommendations using NMF factorization"""
    
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(), None, "User not found"
    
    # Get user index
    user_idx = user_item_matrix.index.get_loc(user_id)
    
    # Reconstruct ratings for this user
    user_predictions = W[user_idx] @ H
    
    # Denormalize predictions back to original scale
    user_predictions = scaler.inverse_transform(user_predictions.reshape(1, -1)).flatten()
    
    # Get actual ratings for filtering
    actual_ratings = user_item_matrix.iloc[user_idx].values
    
    # Create recommendations dataframe
    predictions_df = pd.DataFrame({
        'item': user_item_matrix.columns,
        'predicted_rating': user_predictions,
        'actual_rating': actual_ratings
    })
    
    # Filter out already rated items
    unrated = predictions_df[predictions_df['actual_rating'] == 0]
    
    # Get top recommendations
    recommendations = unrated.nlargest(n_recommendations, 'predicted_rating')
    
    # Add course information
    recommendations = recommendations.merge(
        courses[['COURSE_ID', 'TITLE']], 
        left_on='item', 
        right_on='COURSE_ID'
    )
    
    # Add actual statistics from all users
    course_stats = ratings.groupby('item').agg({
        'rating': ['mean', 'count', 'std']
    }).round(2)
    course_stats.columns = ['avg_rating', 'num_ratings', 'rating_std']
    course_stats = course_stats.reset_index()
    
    recommendations = recommendations.merge(course_stats, on='item', how='left')
    recommendations['predicted_rating'] = recommendations['predicted_rating'].round(2)
    
    # Get user's latent features for analysis
    user_features = W[user_idx]
    
    return recommendations, user_features, None

# Analyze latent features
def analyze_latent_features(W, H, user_item_matrix, courses_df, n_top=5):
    """Analyze what each latent feature represents"""
    
    feature_analysis = []
    genre_columns = [col for col in courses_df.columns if col not in ['COURSE_ID', 'TITLE']]
    
    for feature_idx in range(min(H.shape[0], 10)):  # Limit to first 10 features
        # Get top items for this feature
        top_items_idx = np.argsort(H[feature_idx])[-n_top:][::-1]
        top_items = user_item_matrix.columns[top_items_idx].tolist()
        
        # Analyze genres of top items
        top_courses = courses_df[courses_df['COURSE_ID'].isin(top_items)]
        if not top_courses.empty and genre_columns:
            genre_scores = top_courses[genre_columns].sum()
            dominant_genre = genre_scores.idxmax() if genre_scores.max() > 0 else "Mixed"
        else:
            dominant_genre = "Unknown"
        
        feature_analysis.append({
            'Feature': feature_idx,
            'Dominant Genre': dominant_genre,
            'Strength': np.mean(H[feature_idx])
        })
    
    return pd.DataFrame(feature_analysis)

# Streamlit UI
st.markdown("### How it works")
st.info("""
Non-negative Matrix Factorization (NMF) decomposes the user-item matrix into two lower-rank matrices:
- **W**: User features matrix (users × latent features)
- **H**: Item features matrix (latent features × items)
- Predictions are made by reconstructing the matrix: R ≈ W × H
""")

# Create user-item matrix
with st.spinner("Creating user-item matrix..."):
    user_item_matrix = create_user_item_matrix(ratings)
    matrix_normalized, scaler = prepare_nmf_data(user_item_matrix)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    n_components = st.slider(
        "Number of Latent Features",
        min_value=5,
        max_value=50,
        value=20,
        help="Number of latent features to extract"
    )
    
    max_iter = st.slider(
        "Max Iterations",
        min_value=100,
        max_value=500,
        value=200,
        step=50,
        help="Maximum iterations for NMF convergence"
    )
    
    n_recommendations = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of recommendations to generate"
    )
    
    st.markdown("---")
    st.subheader("📊 Dataset Info")
    st.metric("Total Users", len(user_item_matrix))
    st.metric("Total Items", len(user_item_matrix.columns))
    st.metric("Total Ratings", ratings.shape[0])
    sparsity = 1 - ratings.shape[0] / (len(user_item_matrix) * len(user_item_matrix.columns))
    st.metric("Sparsity", f"{sparsity:.2%}")

# Train NMF model
with st.spinner(f"Training NMF model with {n_components} components..."):
    W, H, nmf_model, reconstruction_error = train_nmf_simple(
        matrix_normalized, 
        n_components=n_components,
        max_iter=max_iter
    )

# Display reconstruction error
st.success(f"✅ Model trained! Reconstruction error: {reconstruction_error:.4f}")

# Main interface tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Get Recommendations", 
    "📊 Latent Features Analysis", 
    "🔍 Model Evaluation",
    "📈 Visualization"
])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Select User")
        
        # User selection
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
                    if len(user_ratings) > 0:
                        st.metric("Avg Rating", f"{user_ratings['rating'].mean():.2f}")
                    else:
                        st.metric("Avg Rating", "N/A")
                
                # Generate recommendations
                if st.button("🚀 Get Recommendations", type="primary"):
                    with st.spinner("Generating recommendations..."):
                        recommendations, user_features, error = get_nmf_recommendations(
                            user_id, 
                            user_item_matrix, 
                            W, 
                            H, 
                            scaler,
                            n_recommendations
                        )
                        
                        st.session_state['recommendations'] = recommendations
                        st.session_state['user_features'] = user_features
                        st.session_state['error'] = error
                        st.session_state['selected_user'] = user_id
    
    with col2:
        st.subheader("📚 Recommendations")
        
        if 'recommendations' in st.session_state:
            if st.session_state.get('error'):
                st.error(st.session_state['error'])
            elif not st.session_state['recommendations'].empty:
                st.success(f"✅ Generated {len(st.session_state['recommendations'])} recommendations")
                
                # Display recommendations
                display_df = st.session_state['recommendations'][
                    ['TITLE', 'predicted_rating', 'avg_rating', 'num_ratings']
                ].copy()
                display_df.columns = ['Course Title', 'Predicted Rating', 'Actual Avg Rating', 'Total Ratings']
                display_df['Actual Avg Rating'] = display_df['Actual Avg Rating'].fillna(0).round(2)
                display_df['Total Ratings'] = display_df['Total Ratings'].fillna(0).astype(int)
                display_df.index = range(1, len(display_df) + 1)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualization
                if 'avg_rating' in st.session_state['recommendations'].columns:
                    fig = px.scatter(
                        st.session_state['recommendations'],
                        x='avg_rating',
                        y='predicted_rating',
                        size='num_ratings',
                        hover_data=['TITLE'],
                        title='NMF Predictions vs Actual Ratings',
                        labels={
                            'avg_rating': 'Actual Average Rating',
                            'predicted_rating': 'NMF Predicted Rating',
                            'num_ratings': 'Number of Ratings'
                        },
                        color='predicted_rating',
                        color_continuous_scale='RdYlGn'
                    )
                    
                    # Add diagonal line
                    fig.add_trace(go.Scatter(
                        x=[0, 5],
                        y=[0, 5],
                        mode='lines',
                        line=dict(dash='dash', color='gray'),
                        showlegend=False
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # User's latent features
                if 'user_features' in st.session_state and st.session_state['user_features'] is not None:
                    st.markdown("#### 🧬 User's Latent Feature Profile")
                    
                    fig = px.bar(
                        x=range(len(st.session_state['user_features'])),
                        y=st.session_state['user_features'],
                        title=f"User {st.session_state.get('selected_user', 'Unknown')}'s Feature Strengths",
                        labels={'x': 'Feature Index', 'y': 'Feature Strength'},
                        color=st.session_state['user_features'],
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📊 Latent Features Analysis")
    
    # Analyze what each latent feature represents
    with st.spinner("Analyzing latent features..."):
        feature_analysis = analyze_latent_features(W, H, user_item_matrix, courses, n_top=5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏷️ Feature Interpretation")
        st.dataframe(feature_analysis, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 📈 Feature Importance")
        
        # Calculate feature importance (variance in H)
        feature_importance = np.var(H, axis=1)
        
        fig = px.bar(
            x=range(min(len(feature_importance), 20)),
            y=feature_importance[:20],
            title="Feature Importance (Variance)",
            labels={'x': 'Feature Index', 'y': 'Variance'},
            color=feature_importance[:20],
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # User-Feature heatmap
    st.markdown("#### 🔥 User-Feature Heatmap (Sample)")
    
    sample_users = min(30, len(W))
    sample_indices = np.random.choice(len(W), sample_users, replace=False)
    
    fig = px.imshow(
        W[sample_indices][:, :min(20, W.shape[1])],  # Limit features shown
        labels=dict(x="Feature", y="User (Sample)", color="Strength"),
        title=f"User-Feature Matrix (Sample of {sample_users} users)",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🔍 Model Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Reconstruction Quality")
        
        # Calculate reconstruction for a sample
        sample_size = min(100, len(user_item_matrix))
        sample_users = np.random.choice(range(len(W)), sample_size, replace=False)
        
        reconstruction_errors = []
        for user_idx in sample_users:
            actual = matrix_normalized.iloc[user_idx].values
            predicted = W[user_idx] @ H
            mask = actual > 0.1  # Only consider rated items
            if mask.any():
                error = np.mean((actual[mask] - predicted[mask]) ** 2)
                reconstruction_errors.append(error)
        
        if reconstruction_errors:
            fig = px.histogram(
                reconstruction_errors,
                nbins=20,
                title="Distribution of Reconstruction Errors",
                labels={'value': 'MSE', 'count': 'Frequency'},
                color_discrete_sequence=['lightblue']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Mean Reconstruction Error", f"{np.mean(reconstruction_errors):.4f}")
            st.metric("Std Reconstruction Error", f"{np.std(reconstruction_errors):.4f}")
    
    with col2:
        st.markdown("#### 📈 Model Metrics")
        
        # Calculate coverage
        total_predictions = W @ H
        coverage = np.sum(total_predictions > 0.5) / total_predictions.size
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Matrix Rank", n_components)
            st.metric("Coverage", f"{coverage:.2%}")
        with col_b:
            st.metric("Training Error", f"{reconstruction_error:.4f}")
            st.metric("Iterations", max_iter)

with tab4:
    st.subheader("📈 Visualization")
    
    # Matrix factorization visualization
    st.markdown("#### 🔄 Matrix Factorization")
    
    # Create a simple visualization of the factorization
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('User-Item Matrix (Sample)', 'User Features (W)', 'Item Features (H)'),
        column_widths=[0.4, 0.3, 0.3]
    )
    
    # Sample for visualization
    sample_u = min(20, len(user_item_matrix))
    sample_i = min(20, len(user_item_matrix.columns))
    
    # Original matrix sample
    fig.add_trace(
        go.Heatmap(
            z=user_item_matrix.iloc[:sample_u, :sample_i].values,
            colorscale='Blues',
            showscale=False
        ),
        row=1, col=1
    )
    
    # W matrix sample
    fig.add_trace(
        go.Heatmap(
            z=W[:sample_u, :min(10, W.shape[1])],
            colorscale='Greens',
            showscale=False
        ),
        row=1, col=2
    )
    
    # H matrix sample
    fig.add_trace(
        go.Heatmap(
            z=H[:min(10, H.shape[0]), :sample_i],
            colorscale='Reds',
            showscale=False
        ),
        row=1, col=3
    )
    
    fig.update_layout(height=400, title_text="NMF Decomposition Visualization")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 💡 Interpretation")
    st.markdown("""
    - **Left**: Original user-item rating matrix (sample)
    - **Middle**: User latent features (W) - each row represents a user's preferences
    - **Right**: Item latent features (H) - each column represents an item's characteristics
    - The product W × H approximates the original matrix
    """)

# Footer
st.markdown("---")
st.markdown("### 💡 Algorithm Details")
with st.expander("Understanding NMF Collaborative Filtering"):
    st.markdown("""
    **Non-negative Matrix Factorization (NMF):**
    
    NMF decomposes the user-item rating matrix R into two non-negative matrices:
    - **W** (users × k): User feature matrix - represents user preferences
    - **H** (k × items): Item feature matrix - represents item characteristics
    - **R ≈ W × H**: Reconstructed ratings
    
    **Key Properties:**
    - All values are non-negative (suitable for ratings)
    - Latent features can be interpreted as topics or preferences
    - Provides a compressed representation of the data
    
    **Advantages:**
    - Fast training and prediction
    - Memory efficient
    - Good for sparse data
    - No negative predictions
    - Interpretable features
    
    **Limitations:**
    - Requires choosing number of components
    - May converge to local optima
    - Assumes linear relationships
    """)
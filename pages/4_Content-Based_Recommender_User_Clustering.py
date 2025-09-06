# pages/4_Content-Based_Recommender_User_Clustering.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="User Clustering Recommender", layout="wide")
st.title("👥 Content-Based Recommender: User Profile Clustering")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    courses = pd.read_csv("data/course_genre.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return courses, ratings

courses, ratings = load_data()

# Create user-feature matrix based on genre preferences
@st.cache_data
def create_user_feature_matrix(ratings_df, courses_df):
    """Create a user-feature matrix based on genre preferences from rated courses"""
    
    # Get genre columns
    genre_columns = [col for col in courses_df.columns if col not in ['COURSE_ID', 'TITLE']]
    
    # Initialize user feature matrix
    users = ratings_df['user'].unique()
    user_feature_matrix = pd.DataFrame(0, index=users, columns=genre_columns)
    
    # For each user, aggregate their genre preferences based on ratings
    for user in users:
        user_ratings = ratings_df[ratings_df['user'] == user]
        
        # Weight genres by ratings
        for _, rating_row in user_ratings.iterrows():
            course_id = rating_row['item']
            rating_value = rating_row['rating']
            
            # Get course genres
            course_data = courses_df[courses_df['COURSE_ID'] == course_id]
            if not course_data.empty:
                for genre in genre_columns:
                    if course_data.iloc[0][genre] == 1:
                        # Add weighted genre preference
                        user_feature_matrix.loc[user, genre] += rating_value
        
        # Normalize by number of ratings to get average preference
        num_ratings = len(user_ratings)
        if num_ratings > 0:
            user_feature_matrix.loc[user] = user_feature_matrix.loc[user] / num_ratings
    
    return user_feature_matrix

# Determine optimal number of clusters
@st.cache_data
def find_optimal_clusters(user_feature_matrix, max_k=10):
    """Find optimal number of clusters using elbow method and silhouette score"""
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(user_feature_matrix)
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, min(max_k + 1, len(user_feature_matrix)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, cluster_labels))
    
    return list(K_range), inertias, silhouette_scores

# Perform clustering
@st.cache_data
def perform_clustering(user_feature_matrix, n_clusters):
    """Perform KMeans clustering on user features"""
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(user_feature_matrix)
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to users
    user_clusters = pd.DataFrame({
        'user': user_feature_matrix.index,
        'cluster': cluster_labels
    })
    
    # Calculate cluster centers in original space
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=user_feature_matrix.columns
    )
    
    return user_clusters, cluster_centers, scaled_features

# Get cluster-based recommendations
def get_cluster_recommendations(user_id, user_clusters, ratings_df, courses_df, n_recommendations=10):
    """Get recommendations based on user's cluster preferences"""
    
    # Find user's cluster
    user_cluster_data = user_clusters[user_clusters['user'] == user_id]
    if user_cluster_data.empty:
        return pd.DataFrame(), None, "User not found"
    
    user_cluster = user_cluster_data['cluster'].values[0]
    
    # Get all users in the same cluster
    cluster_users = user_clusters[user_clusters['cluster'] == user_cluster]['user'].tolist()
    
    # Get all ratings from cluster users
    cluster_ratings = ratings_df[ratings_df['user'].isin(cluster_users)]
    
    # Get user's already rated courses
    user_rated_courses = ratings_df[ratings_df['user'] == user_id]['item'].tolist()
    
    # Aggregate ratings by course for the cluster
    cluster_course_stats = cluster_ratings.groupby('item').agg({
        'rating': ['mean', 'count']
    }).round(2)
    cluster_course_stats.columns = ['avg_rating', 'rating_count']
    cluster_course_stats = cluster_course_stats.reset_index()
    
    # Filter out courses already rated by the user
    recommendations = cluster_course_stats[~cluster_course_stats['item'].isin(user_rated_courses)]
    
    # Filter by minimum rating count (at least 2 ratings from cluster)
    recommendations = recommendations[recommendations['rating_count'] >= 2]
    
    # Sort by average rating and then by count
    recommendations = recommendations.sort_values(['avg_rating', 'rating_count'], ascending=[False, False])
    recommendations = recommendations.head(n_recommendations)
    
    # Add course titles and genres
    recommendations = recommendations.merge(
        courses_df[['COURSE_ID', 'TITLE']], 
        left_on='item', 
        right_on='COURSE_ID'
    )
    
    # Add genre information
    genre_columns = [col for col in courses_df.columns if col not in ['COURSE_ID', 'TITLE']]
    for _, rec in recommendations.iterrows():
        course_genres = []
        course_data = courses_df[courses_df['COURSE_ID'] == rec['item']]
        if not course_data.empty:
            for genre in genre_columns:
                if course_data.iloc[0][genre] == 1:
                    course_genres.append(genre)
        rec['genres'] = ', '.join(course_genres) if course_genres else 'No genres'
    
    return recommendations, user_cluster, None

# Streamlit UI
st.markdown("### How it works")
st.info("""
This recommender groups users into clusters based on their genre preferences derived from their ratings.
Users in the same cluster have similar tastes, and recommendations come from popular courses within your cluster.
""")

# Create user feature matrix
with st.spinner("Creating user preference profiles..."):
    user_feature_matrix = create_user_feature_matrix(ratings, courses)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Clustering analysis
    if st.checkbox("Show Clustering Analysis", value=False):
        st.subheader("📊 Optimal Clusters Analysis")
        with st.spinner("Analyzing optimal clusters..."):
            K_range, inertias, silhouette_scores = find_optimal_clusters(user_feature_matrix)
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Elbow Method', 'Silhouette Score')
            )
            
            fig.add_trace(
                go.Scatter(x=K_range, y=inertias, mode='lines+markers', name='Inertia'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=K_range, y=silhouette_scores, mode='lines+markers', name='Silhouette', line=dict(color='orange')),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Number of Clusters", row=1, col=1)
            fig.update_xaxes(title_text="Number of Clusters", row=1, col=2)
            fig.update_yaxes(title_text="Inertia", row=1, col=1)
            fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
            fig.update_layout(height=300, showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Suggest optimal k
            optimal_k = K_range[np.argmax(silhouette_scores)]
            st.success(f"Suggested optimal clusters: {optimal_k}")
    
    n_clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=min(10, len(user_feature_matrix) // 5),
        value=min(5, len(user_feature_matrix) // 10),
        help="Number of user clusters to create"
    )
    
    n_recommendations = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of courses to recommend"
    )
    
    min_cluster_ratings = st.slider(
        "Min Ratings in Cluster",
        min_value=1,
        max_value=10,
        value=2,
        help="Minimum ratings required from cluster members"
    )

# Perform clustering
with st.spinner("Clustering users..."):
    user_clusters, cluster_centers, scaled_features = perform_clustering(user_feature_matrix, n_clusters)

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Get Recommendations", "📊 Cluster Analysis", "🌐 Cluster Visualization", "👥 User Distribution"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Select User")
        
        # User selection with search
        search_user = st.text_input("Search User ID:", "")
        
        if search_user:
            filtered_users = [u for u in user_clusters['user'].tolist() if str(search_user) in str(u)]
        else:
            filtered_users = user_clusters['user'].tolist()
        
        if filtered_users:
            user_id = st.selectbox(
                "Select a User ID",
                options=filtered_users,
                help="Choose a user to generate recommendations"
            )
            
            # Show user info
            if user_id:
                user_cluster_info = user_clusters[user_clusters['user'] == user_id]
                cluster_id = user_cluster_info['cluster'].values[0]
                
                st.markdown("#### 👤 User Information")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("User Cluster", f"Cluster {cluster_id}")
                with col_b:
                    cluster_size = len(user_clusters[user_clusters['cluster'] == cluster_id])
                    st.metric("Cluster Size", f"{cluster_size} users")
                
                # User's rating history
                user_ratings = ratings[ratings['user'] == user_id]
                st.metric("Total Ratings", len(user_ratings))
                st.metric("Avg Rating Given", f"{user_ratings['rating'].mean():.2f}")
                
                # Generate recommendations
                if st.button("🚀 Get Recommendations", type="primary"):
                    recommendations, user_cluster, error = get_cluster_recommendations(
                        user_id, user_clusters, ratings, courses, n_recommendations
                    )
                    
                    st.session_state['recommendations'] = recommendations
                    st.session_state['user_cluster'] = user_cluster
                    st.session_state['error'] = error
                    st.session_state['selected_user'] = user_id
    
    with col2:
        st.subheader("📚 Recommendations")
        
        if 'recommendations' in st.session_state:
            if st.session_state.get('error'):
                st.error(st.session_state['error'])
            elif not st.session_state['recommendations'].empty:
                st.success(f"✅ Found {len(st.session_state['recommendations'])} recommendations from Cluster {st.session_state['user_cluster']}")
                
                # Display recommendations
                display_df = st.session_state['recommendations'][['TITLE', 'avg_rating', 'rating_count']].copy()
                display_df.columns = ['Course Title', 'Cluster Avg Rating', 'Cluster Ratings Count']
                display_df['Cluster Avg Rating'] = display_df['Cluster Avg Rating'].round(2)
                display_df.index = range(1, len(display_df) + 1)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualization of recommendations
                fig = px.scatter(
                    st.session_state['recommendations'],
                    x='rating_count',
                    y='avg_rating',
                    size='rating_count',
                    hover_data=['TITLE'],
                    title='Recommended Courses from Your Cluster',
                    labels={'rating_count': 'Number of Cluster Ratings', 'avg_rating': 'Average Rating'},
                    color='avg_rating',
                    color_continuous_scale='RdYlGn',
                    size_max=30
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No recommendations found. Try adjusting the minimum ratings threshold.")

with tab2:
    st.subheader("📊 Cluster Characteristics")
    
    # Analyze cluster centers
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster sizes
        cluster_sizes = user_clusters['cluster'].value_counts().sort_index()
        fig = px.bar(
            x=cluster_sizes.index,
            y=cluster_sizes.values,
            title="Users per Cluster",
            labels={'x': 'Cluster ID', 'y': 'Number of Users'},
            color=cluster_sizes.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top genres per cluster
        st.markdown("#### 🏷️ Top Genres by Cluster")
        for cluster_id in range(n_clusters):
            top_genres = cluster_centers.iloc[cluster_id].nlargest(3)
            st.markdown(f"**Cluster {cluster_id}:** {', '.join(top_genres.index.tolist())}")
    
    # Heatmap of cluster centers
    st.subheader("🔥 Cluster Genre Preferences Heatmap")
    
    fig = px.imshow(
        cluster_centers.T,
        labels=dict(x="Cluster", y="Genre", color="Preference Strength"),
        title="Genre Preferences by Cluster",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster statistics
    st.subheader("📈 Cluster Statistics")
    
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_users_list = user_clusters[user_clusters['cluster'] == cluster_id]['user'].tolist()
        cluster_ratings = ratings[ratings['user'].isin(cluster_users_list)]
        
        cluster_stats.append({
            'Cluster': cluster_id,
            'Users': len(cluster_users_list),
            'Total Ratings': len(cluster_ratings),
            'Avg Ratings/User': len(cluster_ratings) / len(cluster_users_list) if cluster_users_list else 0,
            'Avg Rating Given': cluster_ratings['rating'].mean() if not cluster_ratings.empty else 0
        })
    
    cluster_stats_df = pd.DataFrame(cluster_stats)
    cluster_stats_df = cluster_stats_df.round(2)
    st.dataframe(cluster_stats_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("🌐 Cluster Visualization")
    
    # PCA for visualization
    if scaled_features.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(scaled_features)
        
        # Create visualization dataframe
        viz_df = pd.DataFrame({
            'PC1': pca_features[:, 0],
            'PC2': pca_features[:, 1],
            'Cluster': user_clusters['cluster'].values,
            'User': user_clusters['user'].values
        })
        
        # Scatter plot
        fig = px.scatter(
            viz_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            title='User Clusters (PCA Visualization)',
            hover_data=['User'],
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Add cluster centers
        pca_centers = pca.transform(StandardScaler().fit_transform(cluster_centers))
        for i, center in enumerate(pca_centers):
            fig.add_trace(go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode='markers',
                marker=dict(size=20, color='black', symbol='x'),
                name=f'Center {i}',
                showlegend=False
            ))
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"PCA explains {pca.explained_variance_ratio_.sum():.1%} of the variance")
    else:
        st.warning("Not enough dimensions for PCA visualization")
    
    # 3D visualization if possible
    if scaled_features.shape[1] > 2 and st.checkbox("Show 3D Visualization"):
        pca_3d = PCA(n_components=3, random_state=42)
        pca_3d_features = pca_3d.fit_transform(scaled_features)
        
        viz_3d_df = pd.DataFrame({
            'PC1': pca_3d_features[:, 0],
            'PC2': pca_3d_features[:, 1],
            'PC3': pca_3d_features[:, 2],
            'Cluster': user_clusters['cluster'].values,
            'User': user_clusters['user'].values
        })
        
        fig_3d = px.scatter_3d(
            viz_3d_df,
            x='PC1',
            y='PC2',
            z='PC3',
            color='Cluster',
            title='User Clusters (3D PCA)',
            hover_data=['User'],
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_3d.update_layout(height=600)
        st.plotly_chart(fig_3d, use_container_width=True)

with tab4:
    st.subheader("👥 User Distribution Analysis")
    
    # User activity by cluster
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution by cluster
        cluster_rating_dist = []
        for cluster_id in range(n_clusters):
            cluster_users_list = user_clusters[user_clusters['cluster'] == cluster_id]['user'].tolist()
            cluster_ratings = ratings[ratings['user'].isin(cluster_users_list)]
            
            for rating_val in range(1, 6):
                count = len(cluster_ratings[cluster_ratings['rating'] == rating_val])
                cluster_rating_dist.append({
                    'Cluster': f'Cluster {cluster_id}',
                    'Rating': rating_val,
                    'Count': count
                })
        
        rating_dist_df = pd.DataFrame(cluster_rating_dist)
        
        fig = px.bar(
            rating_dist_df,
            x='Rating',
            y='Count',
            color='Cluster',
            title='Rating Distribution by Cluster',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # User activity levels by cluster
        user_activity = ratings.groupby('user')['rating'].count().reset_index()
        user_activity.columns = ['user', 'num_ratings']
        user_activity = user_activity.merge(user_clusters, on='user')
        
        fig = px.box(
            user_activity,
            x='cluster',
            y='num_ratings',
            title='User Activity Distribution by Cluster',
            labels={'cluster': 'Cluster', 'num_ratings': 'Number of Ratings'},
            color='cluster',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Most popular courses by cluster
    st.subheader("🏆 Top Courses by Cluster")
    
    selected_cluster_analysis = st.selectbox("Select Cluster for Analysis", range(n_clusters))
    
    cluster_users_for_analysis = user_clusters[user_clusters['cluster'] == selected_cluster_analysis]['user'].tolist()
    cluster_ratings_analysis = ratings[ratings['user'].isin(cluster_users_for_analysis)]
    
    if not cluster_ratings_analysis.empty:
        top_courses = cluster_ratings_analysis.groupby('item').agg({
            'rating': ['mean', 'count']
        }).round(2)
        top_courses.columns = ['avg_rating', 'rating_count']
        top_courses = top_courses.sort_values(['avg_rating', 'rating_count'], ascending=[False, False]).head(10)
        top_courses = top_courses.merge(courses[['COURSE_ID', 'TITLE']], left_index=True, right_on='COURSE_ID')
        
        fig = px.bar(
            top_courses,
            x='avg_rating',
            y='TITLE',
            orientation='h',
            title=f'Top Courses in Cluster {selected_cluster_analysis}',
            labels={'avg_rating': 'Average Rating', 'TITLE': 'Course'},
            color='rating_count',
            color_continuous_scale='Viridis',
            hover_data=['rating_count']
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 💡 Algorithm Insights")
with st.expander("Understanding User Clustering"):
    st.markdown("""
    **How User Clustering Works:**
    
    1. **User Profile Creation:**
       - Each user's ratings are analyzed to understand their genre preferences
       - Genres are weighted by the ratings given to courses in those genres
       - Creates a preference vector for each user
    
    2. **K-Means Clustering:**
       - Groups users with similar preference vectors
       - Number of clusters can be optimized using elbow method and silhouette score
       - Each cluster represents a group of users with similar tastes
    
    3. **Recommendation Generation:**
       - Identifies the user's cluster
       - Finds popular and highly-rated courses within that cluster
       - Filters out courses the user has already rated
       - Returns top recommendations based on cluster preferences
    
    **Advantages:**
    - Reduces noise by aggregating similar users
    - Can find patterns in user behavior
    - Scalable to large user bases
    - Provides interpretable user segments
    
    **Limitations:**
    - Requires sufficient user activity data
    - Fixed number of clusters may not capture all nuances
    - Cold start problem for new users
    """)
# pages/2_Content-Based_Recommender_User_Profile.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Content-Based Recommender", layout="wide")
st.title("🎯 Content-Based Recommender: User Profile & Course Genres")
st.markdown("---")

# Load your datasets
@st.cache_data
def load_data():
    courses = pd.read_csv("data/course_genre.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return courses, ratings

courses, ratings = load_data()

# Helper function to get course genres as text
def get_course_genres(course_row):
    """Extract genres from a course row as a space-separated string"""
    genre_columns = [col for col in course_row.index if col not in ['COURSE_ID', 'TITLE']]
    genres = [col for col in genre_columns if course_row[col] == 1]
    return ' '.join(genres)

# Create course genre profiles
@st.cache_data
def create_course_profiles(courses_df):
    """Create text profiles for each course based on their genres"""
    courses_df['genre_profile'] = courses_df.apply(get_course_genres, axis=1)
    return courses_df

courses_with_profiles = create_course_profiles(courses)

# Create user profile based on highly rated courses
def create_user_profile(user_id, ratings_df, courses_df, rating_threshold=3.5):
    """Create a user profile based on genres of highly-rated courses"""
    # Get user's highly rated courses
    user_ratings = ratings_df[
        (ratings_df['user'] == user_id) & 
        (ratings_df['rating'] >= rating_threshold)
    ]
    
    if len(user_ratings) == 0:
        return None, []
    
    # Get the courses and their genres
    rated_course_ids = user_ratings['item'].tolist()
    user_courses = courses_df[courses_df['COURSE_ID'].isin(rated_course_ids)]
    
    # Weight genres by ratings
    genre_weights = {}
    genre_columns = [col for col in courses_df.columns if col not in ['COURSE_ID', 'TITLE', 'genre_profile']]
    
    for _, course in user_courses.iterrows():
        course_rating = user_ratings[user_ratings['item'] == course['COURSE_ID']]['rating'].values[0]
        for genre in genre_columns:
            if course[genre] == 1:
                if genre not in genre_weights:
                    genre_weights[genre] = 0
                genre_weights[genre] += course_rating
    
    # Create weighted profile string
    user_profile = ' '.join([
        f"{genre} " * int(weight) 
        for genre, weight in genre_weights.items()
    ])
    
    return user_profile, rated_course_ids

# Content-based recommendation function
def get_content_based_recommendations(user_id, ratings_df, courses_df, n_recommendations=10):
    """Generate content-based recommendations for a user"""
    
    # Create user profile
    user_profile, rated_courses = create_user_profile(user_id, ratings_df, courses_df)
    
    if user_profile is None:
        return pd.DataFrame(), "User has no ratings or no highly-rated courses"
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words=None, token_pattern=r'\b\w+\b')
    
    # Fit on all course profiles plus user profile
    all_profiles = courses_df['genre_profile'].tolist() + [user_profile]
    tfidf_matrix = tfidf.fit_transform(all_profiles)
    
    # User profile is the last vector
    user_vector = tfidf_matrix[-1]
    course_vectors = tfidf_matrix[:-1]
    
    # Calculate cosine similarity
    similarities = cosine_similarity(user_vector, course_vectors).flatten()
    
    # Get recommendations (excluding already rated courses)
    course_similarities = pd.DataFrame({
        'COURSE_ID': courses_df['COURSE_ID'],
        'TITLE': courses_df['TITLE'],
        'similarity': similarities,
        'genres': courses_df['genre_profile']
    })
    
    # Filter out already rated courses
    recommendations = course_similarities[~course_similarities['COURSE_ID'].isin(rated_courses)]
    recommendations = recommendations.nlargest(n_recommendations, 'similarity')
    recommendations['similarity'] = recommendations['similarity'].round(4)
    
    return recommendations, None

# Streamlit UI
st.markdown("### How it works")
st.info("""
This recommender system creates a user profile based on the genres of courses they've rated highly (≥ 3.5 stars). 
It then uses TF-IDF vectorization and cosine similarity to find courses with similar genre profiles.
""")

# Create two columns for the interface
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Configuration")
    
    # User selection
    user_id = st.selectbox(
        "Select a User ID",
        options=sorted(ratings['user'].unique()),
        help="Choose a user to generate recommendations for"
    )
    
    # Number of recommendations
    n_recs = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        help="How many course recommendations to generate"
    )
    
    # Rating threshold
    rating_threshold = st.slider(
        "Rating Threshold for Profile",
        min_value=1.0,
        max_value=5.0,
        value=3.5,
        step=0.5,
        help="Minimum rating to include a course in user profile"
    )
    
    # Generate recommendations button
    if st.button("🚀 Generate Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            # Get user's rating history
            user_ratings = ratings[ratings['user'] == user_id].merge(
                courses[['COURSE_ID', 'TITLE']], 
                left_on='item', 
                right_on='COURSE_ID'
            )
            
            # Store in session state
            st.session_state['user_ratings'] = user_ratings
            st.session_state['recommendations'], error = get_content_based_recommendations(
                user_id, ratings, courses_with_profiles, n_recs
            )
            st.session_state['error'] = error

with col2:
    st.subheader("📊 Results")
    
    # Display user's rating history
    if 'user_ratings' in st.session_state and not st.session_state['user_ratings'].empty:
        with st.expander(f"👤 User {user_id}'s Rating History", expanded=True):
            # Summary stats
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Ratings", len(st.session_state['user_ratings']))
            with col_b:
                st.metric("Avg Rating Given", f"{st.session_state['user_ratings']['rating'].mean():.2f}")
            with col_c:
                high_rated = len(st.session_state['user_ratings'][st.session_state['user_ratings']['rating'] >= rating_threshold])
                st.metric("Highly Rated", high_rated)
            
            # Show rating history
            history_df = st.session_state['user_ratings'][['TITLE', 'rating']].sort_values('rating', ascending=False)
            history_df.columns = ['Course Title', 'Rating']
            st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    # Display recommendations
    if 'recommendations' in st.session_state:
        if 'error' in st.session_state and st.session_state['error']:
            st.warning(st.session_state['error'])
        elif not st.session_state['recommendations'].empty:
            st.success(f"✅ Generated {len(st.session_state['recommendations'])} recommendations for User {user_id}")
            
            # Display recommendations
            recs_display = st.session_state['recommendations'][['TITLE', 'similarity', 'genres']].copy()
            recs_display.columns = ['Course Title', 'Similarity Score', 'Genres']
            recs_display['Similarity Score'] = recs_display['Similarity Score'].apply(lambda x: f"{x:.4f}")
            recs_display.index = range(1, len(recs_display) + 1)
            
            st.dataframe(
                recs_display,
                use_container_width=True,
                column_config={
                    "Course Title": st.column_config.TextColumn(width="medium"),
                    "Similarity Score": st.column_config.TextColumn(width="small"),
                    "Genres": st.column_config.TextColumn(width="medium")
                }
            )
            
            # Visualization of recommendation scores
            st.subheader("📈 Recommendation Strength")
            fig = px.bar(
                st.session_state['recommendations'].head(10),
                x='similarity',
                y='TITLE',
                orientation='h',
                title=f"Top 10 Recommendations by Similarity Score",
                labels={'similarity': 'Similarity Score', 'TITLE': 'Course'},
                color='similarity',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                showlegend=False,
                yaxis={'categoryorder':'total ascending'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

# Add analysis section
st.markdown("---")
st.subheader("🔍 User Profile Analysis")

col1, col2 = st.columns(2)

with col1:
    if st.button("Analyze User Profile"):
        user_profile, rated_courses = create_user_profile(user_id, ratings, courses_with_profiles, rating_threshold)
        
        if user_profile:
            # Count genre frequencies
            genre_columns = [col for col in courses.columns if col not in ['COURSE_ID', 'TITLE', 'genre_profile']]
            user_rated = courses[courses['COURSE_ID'].isin(rated_courses)]
            
            genre_counts = {}
            for genre in genre_columns:
                count = user_rated[genre].sum()
                if count > 0:
                    genre_counts[genre] = count
            
            if genre_counts:
                # Create pie chart
                fig = px.pie(
                    values=list(genre_counts.values()),
                    names=list(genre_counts.keys()),
                    title=f"User {user_id}'s Genre Preferences",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No genre preferences found for this user")
        else:
            st.warning("User has no ratings to analyze")

with col2:
    if st.button("Compare with Popular Genres"):
        # Get overall genre distribution
        genre_columns = [col for col in courses.columns if col not in ['COURSE_ID', 'TITLE']]
        overall_genres = courses[genre_columns].sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=overall_genres.values,
            y=overall_genres.index,
            orientation='h',
            title="Overall Genre Popularity",
            labels={'x': 'Number of Courses', 'y': 'Genre'},
            color=overall_genres.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            showlegend=False,
            yaxis={'categoryorder':'total ascending'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer with explanation
st.markdown("---")
st.markdown("### 💡 Algorithm Details")
with st.expander("How the Content-Based Filtering Works"):
    st.markdown("""
    **1. User Profile Creation:**
    - Identifies courses the user rated ≥ threshold (default 3.5)
    - Extracts genres from these highly-rated courses
    - Weights genres by the actual rating values
    
    **2. TF-IDF Vectorization:**
    - Converts genre profiles into numerical vectors
    - Captures the importance of each genre
    
    **3. Similarity Calculation:**
    - Computes cosine similarity between user profile and all courses
    - Higher similarity = better match with user preferences
    
    **4. Recommendation Generation:**
    - Ranks courses by similarity score
    - Filters out already-rated courses
    - Returns top N recommendations
    
    **Advantages:**
    - No cold start problem for new courses
    - Transparent and interpretable
    - Works well with sparse data
    
    **Limitations:**
    - Limited to genre-based features
    - May lack diversity in recommendations
    - Doesn't consider collaborative signals
    """)
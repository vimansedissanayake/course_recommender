# pages/3_Content-Based_Recommender_Course_Similarity.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

st.set_page_config(page_title="Course Similarity Recommender", layout="wide")
st.title("🔗 Content-Based Recommender: Course Similarity")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    courses = pd.read_csv("data/course_genre.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return courses, ratings

courses, ratings = load_data()

# Create course genre profiles
@st.cache_data
def prepare_course_features(courses_df):
    """Prepare course features for similarity computation"""
    # Get genre columns
    genre_columns = [col for col in courses_df.columns if col not in ['COURSE_ID', 'TITLE']]
    
    # Create text representation of genres for each course
    course_genres = []
    for _, row in courses_df.iterrows():
        genres = ' '.join([col for col in genre_columns if row[col] == 1])
        # Repeat genre names to give them more weight
        genres = ' '.join([genres] * 3)  # Triple weight for genres
        course_genres.append(genres)
    
    courses_df['genre_text'] = course_genres
    
    # Add course title tokens (for title similarity)
    courses_df['combined_features'] = courses_df['TITLE'] + ' ' + courses_df['genre_text']
    
    return courses_df

courses_featured = prepare_course_features(courses)

# Compute similarity matrix
@st.cache_data
def compute_similarity_matrix(courses_df, method='tfidf'):
    """Compute similarity matrix between courses"""
    
    if method == 'tfidf':
        # TF-IDF based similarity
        tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            max_features=100
        )
        tfidf_matrix = tfidf.fit_transform(courses_df['combined_features'])
        similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
        
    elif method == 'genre_jaccard':
        # Jaccard similarity based on genres
        genre_columns = [col for col in courses_df.columns 
                        if col not in ['COURSE_ID', 'TITLE', 'genre_text', 'combined_features']]
        genre_matrix = courses_df[genre_columns].values
        
        # Compute Jaccard similarity
        similarity_matrix = np.zeros((len(courses_df), len(courses_df)))
        for i in range(len(courses_df)):
            for j in range(len(courses_df)):
                intersection = np.sum(genre_matrix[i] * genre_matrix[j])
                union = np.sum(np.logical_or(genre_matrix[i], genre_matrix[j]))
                similarity_matrix[i][j] = intersection / union if union > 0 else 0
                
    elif method == 'genre_cosine':
        # Cosine similarity based on genre binary vectors
        genre_columns = [col for col in courses_df.columns 
                        if col not in ['COURSE_ID', 'TITLE', 'genre_text', 'combined_features']]
        genre_matrix = courses_df[genre_columns].values
        similarity_matrix = cosine_similarity(genre_matrix)
    
    return similarity_matrix

# Get recommendations function
def get_recommendations(course_title, courses_df, similarity_matrix, n_recommendations=10):
    """Get course recommendations based on similarity"""
    
    # Get the index of the course
    try:
        idx = courses_df[courses_df['TITLE'] == course_title].index[0]
    except IndexError:
        return pd.DataFrame(), "Course not found"
    
    # Get similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort by similarity (excluding the course itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
    
    # Get course indices
    course_indices = [i[0] for i in sim_scores]
    
    # Create recommendations dataframe
    recommendations = courses_df.iloc[course_indices][['COURSE_ID', 'TITLE']].copy()
    recommendations['similarity_score'] = [score[1] for score in sim_scores]
    recommendations['similarity_percentage'] = (recommendations['similarity_score'] * 100).round(1)
    
    # Add genre information
    genre_columns = [col for col in courses_df.columns 
                    if col not in ['COURSE_ID', 'TITLE', 'genre_text', 'combined_features']]
    
    # Get common genres with selected course
    selected_genres = courses_df.iloc[idx][genre_columns]
    common_genres = []
    for idx_rec in course_indices:
        rec_genres = courses_df.iloc[idx_rec][genre_columns]
        common = [genre for genre in genre_columns 
                 if selected_genres[genre] == 1 and rec_genres[genre] == 1]
        common_genres.append(', '.join(common) if common else 'No common genres')
    
    recommendations['common_genres'] = common_genres
    
    # Add popularity metrics if available
    course_ratings = ratings.groupby('item').agg({
        'rating': ['count', 'mean']
    }).round(2)
    course_ratings.columns = ['num_ratings', 'avg_rating']
    course_ratings = course_ratings.reset_index()
    
    recommendations = recommendations.merge(
        course_ratings, 
        left_on='COURSE_ID', 
        right_on='item', 
        how='left'
    )
    recommendations = recommendations.drop('item', axis=1, errors='ignore')
    recommendations['num_ratings'] = recommendations['num_ratings'].fillna(0).astype(int)
    recommendations['avg_rating'] = recommendations['avg_rating'].fillna(0).round(2)
    
    return recommendations, None

# Streamlit UI
st.markdown("### How it works")
st.info("""
This recommender finds courses similar to a selected course based on:
- **Genre similarity**: Courses with similar genre combinations
- **Title similarity**: Courses with similar titles
- **Multiple similarity metrics**: TF-IDF, Jaccard, or Cosine similarity
""")

# Configuration sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    similarity_method = st.selectbox(
        "Similarity Method",
        options=['tfidf', 'genre_jaccard', 'genre_cosine'],
        format_func=lambda x: {
            'tfidf': 'TF-IDF (Title + Genres)',
            'genre_jaccard': 'Jaccard (Genres Only)',
            'genre_cosine': 'Cosine (Genres Only)'
        }[x],
        help="Choose the similarity calculation method"
    )
    
    n_recommendations = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        help="How many similar courses to show"
    )
    
    show_network = st.checkbox(
        "Show Network Visualization",
        value=False,
        help="Display course similarity network"
    )

# Compute similarity matrix
similarity_matrix = compute_similarity_matrix(courses_featured, method=similarity_method)

# Main interface
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 Select a Course")
    
    # Add search functionality
    search_term = st.text_input("🔍 Search course by title:", "")
    
    if search_term:
        filtered_courses = courses_featured[
            courses_featured['TITLE'].str.contains(search_term, case=False, na=False)
        ]['TITLE'].tolist()
    else:
        filtered_courses = courses_featured['TITLE'].tolist()
    
    if filtered_courses:
        course_title = st.selectbox(
            "Select a Course",
            options=filtered_courses,
            help="Choose a course to find similar ones"
        )
        
        # Display selected course info
        if course_title:
            selected_course = courses_featured[courses_featured['TITLE'] == course_title].iloc[0]
            
            st.markdown("#### 📚 Selected Course Details")
            
            # Get genres for this course
            genre_columns = [col for col in courses.columns 
                            if col not in ['COURSE_ID', 'TITLE']]
            course_genres = [col for col in genre_columns if selected_course[col] == 1]
            
            st.markdown(f"**Course ID:** {selected_course['COURSE_ID']}")
            st.markdown(f"**Genres:** {', '.join(course_genres) if course_genres else 'No genres'}")
            
            # Get rating statistics
            course_ratings = ratings[ratings['item'] == selected_course['COURSE_ID']]
            if not course_ratings.empty:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Avg Rating", f"{course_ratings['rating'].mean():.2f}")
                with col_b:
                    st.metric("Total Ratings", len(course_ratings))
            
            # Generate recommendations button
            if st.button("🔍 Find Similar Courses", type="primary"):
                st.session_state['recommendations'], error = get_recommendations(
                    course_title, 
                    courses_featured, 
                    similarity_matrix, 
                    n_recommendations
                )
                st.session_state['selected_course'] = course_title
                st.session_state['error'] = error
    else:
        st.warning("No courses found matching your search term.")

with col2:
    st.subheader("📊 Similar Courses")
    
    if 'recommendations' in st.session_state:
        if st.session_state.get('error'):
            st.error(st.session_state['error'])
        elif not st.session_state['recommendations'].empty:
            st.success(f"✅ Found {len(st.session_state['recommendations'])} similar courses to '{st.session_state['selected_course']}'")
            
            # Display recommendations table
            display_df = st.session_state['recommendations'][
                ['TITLE', 'similarity_percentage', 'common_genres', 'avg_rating', 'num_ratings']
            ].copy()
            display_df.columns = ['Course Title', 'Similarity %', 'Common Genres', 'Avg Rating', '# Ratings']
            display_df.index = range(1, len(display_df) + 1)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "Course Title": st.column_config.TextColumn(width="large"),
                    "Similarity %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Common Genres": st.column_config.TextColumn(width="medium"),
                    "Avg Rating": st.column_config.NumberColumn(format="%.2f ⭐"),
                    "# Ratings": st.column_config.NumberColumn()
                }
            )
            
            # Visualization tabs
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Similarity Scores", "🏷️ Genre Overlap", "⭐ Rating Comparison"])
            
            with viz_tab1:
                # Bar chart of similarity scores
                fig = px.bar(
                    st.session_state['recommendations'].head(10),
                    x='similarity_percentage',
                    y='TITLE',
                    orientation='h',
                    title=f"Top 10 Most Similar Courses",
                    labels={'similarity_percentage': 'Similarity %', 'TITLE': 'Course'},
                    color='similarity_percentage',
                    color_continuous_scale='Viridis',
                    text='similarity_percentage'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(
                    showlegend=False,
                    yaxis={'categoryorder':'total ascending'},
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab2:
                # Genre overlap visualization
                selected_idx = courses_featured[courses_featured['TITLE'] == st.session_state['selected_course']].index[0]
                genre_columns = [col for col in courses.columns 
                               if col not in ['COURSE_ID', 'TITLE']]
                
                selected_genres = set([col for col in genre_columns 
                                      if courses_featured.iloc[selected_idx][col] == 1])
                
                overlap_data = []
                for _, rec in st.session_state['recommendations'].head(5).iterrows():
                    rec_idx = courses_featured[courses_featured['TITLE'] == rec['TITLE']].index[0]
                    rec_genres = set([col for col in genre_columns 
                                     if courses_featured.iloc[rec_idx][col] == 1])
                    
                    overlap = len(selected_genres.intersection(rec_genres))
                    unique_to_rec = len(rec_genres - selected_genres)
                    
                    overlap_data.append({
                        'Course': rec['TITLE'][:30] + '...' if len(rec['TITLE']) > 30 else rec['TITLE'],
                        'Common Genres': overlap,
                        'Unique Genres': unique_to_rec
                    })
                
                if overlap_data:
                    overlap_df = pd.DataFrame(overlap_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Common Genres',
                        y=overlap_df['Course'],
                        x=overlap_df['Common Genres'],
                        orientation='h',
                        marker_color='lightgreen'
                    ))
                    fig.add_trace(go.Bar(
                        name='Unique Genres',
                        y=overlap_df['Course'],
                        x=overlap_df['Unique Genres'],
                        orientation='h',
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        barmode='stack',
                        title='Genre Overlap with Selected Course',
                        xaxis_title='Number of Genres',
                        yaxis_title='Course',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab3:
                # Rating comparison
                if 'avg_rating' in st.session_state['recommendations'].columns:
                    fig = px.scatter(
                        st.session_state['recommendations'],
                        x='similarity_percentage',
                        y='avg_rating',
                        size='num_ratings',
                        hover_data=['TITLE'],
                        title='Similarity vs Rating',
                        labels={
                            'similarity_percentage': 'Similarity %',
                            'avg_rating': 'Average Rating',
                            'num_ratings': 'Number of Ratings'
                        },
                        color='avg_rating',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

# Network visualization (if enabled)
if show_network and 'recommendations' in st.session_state and not st.session_state['recommendations'].empty:
    st.markdown("---")
    st.subheader("🌐 Course Similarity Network")
    
    with st.spinner("Building network visualization..."):
        # Create network graph
        threshold = st.slider("Similarity Threshold for Network", 0.1, 0.9, 0.3)
        
        # Get indices for selected course and recommendations
        selected_idx = courses_featured[courses_featured['TITLE'] == st.session_state['selected_course']].index[0]
        rec_indices = [courses_featured[courses_featured['TITLE'] == title].index[0] 
                      for title in st.session_state['recommendations']['TITLE'].head(10)]
        
        all_indices = [selected_idx] + rec_indices
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        selected_title_short = st.session_state['selected_course'][:20] + '...' if len(st.session_state['selected_course']) > 20 else st.session_state['selected_course']
        
        for idx in all_indices:
            G.add_node(courses_featured.iloc[idx]['TITLE'][:20] + '...' 
                      if len(courses_featured.iloc[idx]['TITLE']) > 20 
                      else courses_featured.iloc[idx]['TITLE'])
        
        # Add edges based on similarity
        for i, idx1 in enumerate(all_indices):
            for idx2 in all_indices[i+1:]:
                sim = similarity_matrix[idx1][idx2]
                if sim > threshold:
                    G.add_edge(
                        courses_featured.iloc[idx1]['TITLE'][:20] + '...' 
                        if len(courses_featured.iloc[idx1]['TITLE']) > 20 
                        else courses_featured.iloc[idx1]['TITLE'],
                        courses_featured.iloc[idx2]['TITLE'][:20] + '...' 
                        if len(courses_featured.iloc[idx2]['TITLE']) > 20 
                        else courses_featured.iloc[idx2]['TITLE'],
                        weight=sim
                    )
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge trace
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=G[edge[0]][edge[1]]['weight']*3, color='#888'),
                hoverinfo='none'
            ))
        
        # Create node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=[node for node in G.nodes()],
            textposition="top center",
            marker=dict(
                size=15,
                color=['red' if node == selected_title_short else 'lightblue' for node in G.nodes()],
                line=dict(width=2, color='white')
            ),
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title=f"Course Similarity Network (threshold > {threshold:.1f})",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# Algorithm comparison
st.markdown("---")
st.subheader("🔬 Algorithm Comparison")

with st.expander("Compare Different Similarity Methods"):
    if st.button("Run Comparison"):
        comparison_results = []
        
        for method in ['tfidf', 'genre_jaccard', 'genre_cosine']:
            sim_matrix = compute_similarity_matrix(courses_featured, method=method)
            
            # Get average similarity scores
            avg_sim = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
            max_sim = np.max(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
            
            comparison_results.append({
                'Method': method.replace('_', ' ').title(),
                'Avg Similarity': avg_sim,
                'Max Similarity': max_sim,
                'Sparsity': np.sum(sim_matrix < 0.1) / sim_matrix.size
            })
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Create comparison visualizations
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Average Similarity', 'Max Similarity', 'Sparsity (% < 0.1)')
        )
        
        fig.add_trace(
            go.Bar(x=comparison_df['Method'], y=comparison_df['Avg Similarity'], marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=comparison_df['Method'], y=comparison_df['Max Similarity'], marker_color='lightgreen'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=comparison_df['Method'], y=comparison_df['Sparsity'], marker_color='lightcoral'),
            row=1, col=3
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(comparison_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 💡 Tips")
st.markdown("""
- **TF-IDF** works best when course titles contain meaningful keywords
- **Jaccard similarity** is good for finding courses with exact genre matches
- **Cosine similarity** works well for finding courses with similar genre profiles
- Courses with higher ratings and more reviews might be safer choices
""")
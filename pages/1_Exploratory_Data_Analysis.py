# pages/1_Exploratory_Data_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA - Course Recommendation System", layout="wide")
st.title("📊 Exploratory Data Analysis")
st.markdown("---")

# Load your datasets
@st.cache_data
def load_data():
    courses = pd.read_csv("data/course_genre.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return courses, ratings

courses, ratings = load_data()

# Create tabs for better organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Dataset Overview", "⭐ Ratings Analysis", "🎯 Course Analysis", "👥 User Behavior", "📈 Enrollment Analysis"])

with tab1:
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Courses", f"{len(courses):,}")
    with col2:
        st.metric("Total Users", f"{ratings['user'].nunique():,}")
    with col3:
        st.metric("Total Ratings", f"{len(ratings):,}")
    with col4:
        st.metric("Avg Rating", f"{ratings['rating'].mean():.2f}")
    
    st.subheader("📋 Course Data Sample")
    st.dataframe(courses.head(10), use_container_width=True)
    
    st.subheader("⭐ Ratings Data Sample")
    st.dataframe(ratings.head(10), use_container_width=True)
    
    # Data quality check
    st.subheader("🔍 Data Quality Check")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Courses Dataset**\n- Shape: {courses.shape}\n- Missing values: {courses.isnull().sum().sum()}")
    with col2:
        st.info(f"**Ratings Dataset**\n- Shape: {ratings.shape}\n- Missing values: {ratings.isnull().sum().sum()}")

with tab2:
    st.header("Rating Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced histogram with better styling
        fig_hist = px.histogram(
            ratings, 
            x="rating",
            nbins=10,
            title="Distribution of Course Ratings",
            labels={"rating": "Rating", "count": "Frequency"},
            color_discrete_sequence=["#3498db"]
        )
        fig_hist.update_layout(
            showlegend=False,
            xaxis_title="Rating Score",
            yaxis_title="Number of Ratings",
            bargap=0.1,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        fig_hist.update_traces(marker_line_color='darkblue', marker_line_width=1.5)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Rating distribution pie chart
        rating_counts = ratings['rating'].value_counts().sort_index()
        fig_pie = px.pie(
            values=rating_counts.values,
            names=rating_counts.index,
            title="Rating Proportion",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Rating statistics
    st.subheader("📈 Rating Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Mean", f"{ratings['rating'].mean():.2f}")
    with col2:
        st.metric("Median", f"{ratings['rating'].median():.2f}")
    with col3:
        st.metric("Mode", f"{ratings['rating'].mode()[0]:.2f}")
    with col4:
        st.metric("Std Dev", f"{ratings['rating'].std():.2f}")
    with col5:
        st.metric("Variance", f"{ratings['rating'].var():.2f}")
    
    # Box plot for rating distribution
    fig_box = px.box(
        ratings, 
        y="rating",
        title="Rating Distribution Box Plot",
        color_discrete_sequence=["#2ecc71"]
    )
    fig_box.update_layout(
        showlegend=False,
        yaxis_title="Rating Score",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.header("Course Analysis")
    
    # Merge ratings with courses to get course titles
    course_ratings = ratings.merge(courses[['COURSE_ID', 'TITLE']], left_on='item', right_on='COURSE_ID', how='left')
    
    # Most rated courses
    course_popularity = course_ratings.groupby(['item', 'TITLE']).agg({
        'rating': ['count', 'mean']
    }).round(2)
    course_popularity.columns = ['rating_count', 'avg_rating']
    course_popularity = course_popularity.reset_index()
    
    # TOP 20 MOST POPULAR COURSES (Changed from 10 to 20)
    st.subheader("🏆 Top 20 Most Popular Courses")
    top_20_rated = course_popularity.nlargest(20, 'rating_count')
    
    fig_top20 = px.bar(
        top_20_rated,
        x='rating_count',
        y='TITLE',
        orientation='h',
        title="Top 20 Most Popular Courses by Number of Ratings",
        labels={'rating_count': 'Number of Ratings', 'TITLE': 'Course Title'},
        color='avg_rating',
        color_continuous_scale='Viridis',
        hover_data={'avg_rating': ':.2f', 'rating_count': True}
    )
    fig_top20.update_layout(
        yaxis={'categoryorder':'total ascending'},
        height=600,
        coloraxis_colorbar_title="Avg Rating",
        xaxis_title="Number of Enrollments/Ratings",
        margin=dict(l=300)  # Increased left margin for longer course titles
    )
    st.plotly_chart(fig_top20, use_container_width=True)
    
    # WORD CLOUD SECTION
    st.subheader("☁️ Course Titles Word Cloud")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create word cloud from course titles
        # Weight words by their popularity (rating count)
        course_text_data = []
        for _, row in course_popularity.iterrows():
            # Repeat title based on rating count (normalized)
            repeat_count = int(row['rating_count'] / 10) + 1  # Scale down for performance
            course_text_data.extend([row['TITLE']] * repeat_count)
        
        text = ' '.join(course_text_data)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        # Display word cloud
        fig_wc, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Most Frequent Words in Course Titles (Size = Popularity)", fontsize=14, pad=20)
        st.pyplot(fig_wc)
    
    with col2:
        st.info("""
        **Word Cloud Insights:**
        - Larger words appear more frequently in popular courses
        - This visualization helps identify trending topics
        - Common keywords can guide course categorization
        """)
    
    # Additional course analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 highest rated courses (with minimum ratings threshold)
        min_ratings = st.slider("Minimum ratings required", 5, 50, 10)
        filtered_courses = course_popularity[course_popularity['rating_count'] >= min_ratings]
        top_quality = filtered_courses.nlargest(10, 'avg_rating')
        
        fig_bar2 = px.bar(
            top_quality,
            x='avg_rating',
            y='TITLE',
            orientation='h',
            title=f"Top 10 Highest Rated Courses (min {min_ratings} ratings)",
            labels={'avg_rating': 'Average Rating', 'TITLE': 'Course Title'},
            color='rating_count',
            color_continuous_scale='Reds'
        )
        fig_bar2.update_layout(
            yaxis={'categoryorder':'total ascending'},
            height=400,
            coloraxis_colorbar_title="# Ratings"
        )
        st.plotly_chart(fig_bar2, use_container_width=True)
    
    with col2:
        # Course genre distribution
        st.subheader("🏷️ Top Course Genres")
        genre_columns = [col for col in courses.columns if col not in ['COURSE_ID', 'TITLE']]
        genre_counts = courses[genre_columns].sum().sort_values(ascending=False).head(10)
        
        fig_genre = px.bar(
            x=genre_counts.values,
            y=genre_counts.index,
            orientation='h',
            title="Top 10 Course Genres",
            labels={'x': 'Number of Courses', 'y': 'Genre'},
            color=genre_counts.values,
            color_continuous_scale='Turbo'
        )
        fig_genre.update_layout(
            showlegend=False,
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig_genre, use_container_width=True)

with tab4:
    st.header("User Behavior Analysis")
    
    # User activity analysis
    user_activity = ratings.groupby('user').agg({
        'rating': ['count', 'mean', 'std']
    }).round(2)
    user_activity.columns = ['ratings_count', 'avg_rating', 'rating_std']
    user_activity = user_activity.reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of ratings per user
        fig_user_dist = px.histogram(
            user_activity,
            x='ratings_count',
            nbins=30,
            title="Distribution of Ratings per User",
            labels={'ratings_count': 'Number of Ratings', 'count': 'Number of Users'},
            color_discrete_sequence=['#e74c3c']
        )
        fig_user_dist.update_layout(
            xaxis_title="Number of Ratings Given",
            yaxis_title="Number of Users",
            showlegend=False,
            bargap=0.1
        )
        st.plotly_chart(fig_user_dist, use_container_width=True)
    
    with col2:
        # User rating behavior scatter plot
        fig_scatter = px.scatter(
            user_activity[user_activity['ratings_count'] > 1],  # Filter users with multiple ratings
            x='ratings_count',
            y='avg_rating',
            size='rating_std',
            title="User Rating Patterns",
            labels={'ratings_count': 'Number of Ratings', 'avg_rating': 'Average Rating Given'},
            color='avg_rating',
            color_continuous_scale='Bluered',
            hover_data=['user']
        )
        fig_scatter.update_layout(
            xaxis_title="Number of Ratings by User",
            yaxis_title="Average Rating Given",
            coloraxis_colorbar_title="Avg Rating"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # User segments
    st.subheader("👥 User Segments")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        active_users = len(user_activity[user_activity['ratings_count'] >= 10])
        st.metric("Active Users", f"{active_users:,}", f"{active_users/len(user_activity)*100:.1f}%")
    
    with col2:
        moderate_users = len(user_activity[(user_activity['ratings_count'] >= 5) & (user_activity['ratings_count'] < 10)])
        st.metric("Moderate Users", f"{moderate_users:,}", f"{moderate_users/len(user_activity)*100:.1f}%")
    
    with col3:
        casual_users = len(user_activity[user_activity['ratings_count'] < 5])
        st.metric("Casual Users", f"{casual_users:,}", f"{casual_users/len(user_activity)*100:.1f}%")
    
    # Rating patterns over items
    st.subheader("📊 Rating Patterns")
    
    # Create a heatmap showing rating distribution by rating value and user activity level
    user_activity['activity_level'] = pd.cut(
        user_activity['ratings_count'],
        bins=[0, 5, 10, 20, float('inf')],
        labels=['Casual (1-5)', 'Moderate (6-10)', 'Active (11-20)', 'Power (20+)']
    )
    
    # Merge with ratings to get rating distribution by user activity level
    ratings_with_activity = ratings.merge(
        user_activity[['user', 'activity_level']], 
        on='user', 
        how='left'
    )
    
    heatmap_data = ratings_with_activity.groupby(['activity_level', 'rating']).size().unstack(fill_value=0)
    
    fig_heatmap = px.imshow(
        heatmap_data.T,
        title="Rating Distribution by User Activity Level",
        labels=dict(x="User Activity Level", y="Rating", color="Count"),
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# NEW TAB: ENROLLMENT ANALYSIS
with tab5:
    st.header("📈 Enrollment Distribution Analysis")
    
    # Calculate enrollment distribution (number of courses per user)
    user_enrollments = ratings.groupby('user')['item'].nunique().reset_index()
    user_enrollments.columns = ['user', 'courses_enrolled']
    
    # Create enrollment distribution
    enrollment_dist = user_enrollments['courses_enrolled'].value_counts().sort_index().reset_index()
    enrollment_dist.columns = ['num_courses', 'num_users']
    
    # Main enrollment distribution chart
    st.subheader("📊 User Enrollment Distribution")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig_enrollment = px.bar(
            enrollment_dist.head(30),  # Show up to 30 courses
            x='num_courses',
            y='num_users',
            title="Number of Users by Course Enrollment Count",
            labels={'num_courses': 'Number of Courses Enrolled', 'num_users': 'Number of Users'},
            color='num_users',
            color_continuous_scale='Viridis',
            text='num_users'
        )
        fig_enrollment.update_traces(texttemplate='%{text}', textposition='outside')
        fig_enrollment.update_layout(
            height=500,
            showlegend=False,
            xaxis=dict(
                title="Number of Courses Enrolled",
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            yaxis_title="Number of Users"
        )
        st.plotly_chart(fig_enrollment, use_container_width=True)
    
    with col2:
        # Key statistics
        st.metric("Median Enrollments", f"{user_enrollments['courses_enrolled'].median():.0f} courses")
        st.metric("Mean Enrollments", f"{user_enrollments['courses_enrolled'].mean():.1f} courses")
        st.metric("Max Enrollments", f"{user_enrollments['courses_enrolled'].max()} courses")
        st.metric("Single Course Users", f"{enrollment_dist[enrollment_dist['num_courses']==1]['num_users'].values[0]:,}")
    
    # Detailed enrollment breakdown
    st.subheader("📋 Detailed Enrollment Breakdown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Users with 1 course
        single_course = enrollment_dist[enrollment_dist['num_courses']==1]['num_users'].values[0]
        total_users = user_enrollments.shape[0]
        st.info(f"""
        **Single Course Users**
        - Count: {single_course:,}
        - Percentage: {single_course/total_users*100:.1f}%
        """)
    
    with col2:
        # Users with 2-5 courses
        few_courses = enrollment_dist[(enrollment_dist['num_courses']>=2) & (enrollment_dist['num_courses']<=5)]['num_users'].sum()
        st.info(f"""
        **2-5 Courses**
        - Count: {few_courses:,}
        - Percentage: {few_courses/total_users*100:.1f}%
        """)
    
    with col3:
        # Users with 6-10 courses
        moderate_courses = enrollment_dist[(enrollment_dist['num_courses']>=6) & (enrollment_dist['num_courses']<=10)]['num_users'].sum()
        st.info(f"""
        **6-10 Courses**
        - Count: {moderate_courses:,}
        - Percentage: {moderate_courses/total_users*100:.1f}%
        """)
    
    # Additional row for heavy users
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Users with 11-20 courses
        many_courses = enrollment_dist[(enrollment_dist['num_courses']>=11) & (enrollment_dist['num_courses']<=20)]['num_users'].sum()
        st.info(f"""
        **11-20 Courses**
        - Count: {many_courses:,}
        - Percentage: {many_courses/total_users*100:.1f}%
        """)
    
    with col2:
        # Users with >20 courses
        power_users = enrollment_dist[enrollment_dist['num_courses']>20]['num_users'].sum()
        st.info(f"""
        **20+ Courses (Power Users)**
        - Count: {power_users:,}
        - Percentage: {power_users/total_users*100:.1f}%
        """)
    
    with col3:
        # Top enrollers
        top_enrollers = user_enrollments.nlargest(5, 'courses_enrolled')
        st.info(f"""
        **Top 5 Enrollers**
        {chr(10).join([f"- User {row['user']}: {row['courses_enrolled']} courses" for _, row in top_enrollers.iterrows()])}
        """)
    
    # Cumulative distribution
    st.subheader("📈 Cumulative Enrollment Distribution")
    
    enrollment_dist['cumulative_users'] = enrollment_dist['num_users'].cumsum()
    enrollment_dist['cumulative_percentage'] = (enrollment_dist['cumulative_users'] / total_users * 100).round(1)
    
    fig_cumulative = go.Figure()
    
    fig_cumulative.add_trace(go.Scatter(
        x=enrollment_dist['num_courses'].head(30),
        y=enrollment_dist['cumulative_percentage'].head(30),
        mode='lines+markers',
        name='Cumulative %',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8)
    ))
    
    fig_cumulative.update_layout(
        title="Cumulative Distribution of Course Enrollments",
        xaxis_title="Number of Courses Enrolled (or fewer)",
        yaxis_title="Cumulative Percentage of Users (%)",
        height=400,
        hovermode='x unified',
        showlegend=False
    )
    
    # Add horizontal lines for key percentiles
    fig_cumulative.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50%")
    fig_cumulative.add_hline(y=75, line_dash="dash", line_color="orange", annotation_text="75%")
    fig_cumulative.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="90%")
    
    st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # Enrollment patterns table
    st.subheader("📊 Enrollment Statistics Table")
    
    # Create a summary table
    summary_data = []
    ranges = [(1, 1), (2, 5), (6, 10), (11, 15), (16, 20), (21, float('inf'))]
    range_labels = ['1 course', '2-5 courses', '6-10 courses', '11-15 courses', '16-20 courses', '21+ courses']
    
    for (low, high), label in zip(ranges, range_labels):
        if high == float('inf'):
            mask = enrollment_dist['num_courses'] >= low
        else:
            mask = (enrollment_dist['num_courses'] >= low) & (enrollment_dist['num_courses'] <= high)
        
        users_in_range = enrollment_dist[mask]['num_users'].sum()
        percentage = users_in_range / total_users * 100
        
        summary_data.append({
            'Enrollment Range': label,
            'Number of Users': f"{users_in_range:,}",
            'Percentage': f"{percentage:.1f}%",
            'Cumulative %': f"{enrollment_dist[enrollment_dist['num_courses'] <= (high if high != float('inf') else enrollment_dist['num_courses'].max())]['num_users'].sum() / total_users * 100:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Add a footer with summary insights
st.markdown("---")
st.subheader("💡 Key Insights")

# Calculate some key insights
avg_ratings_per_user = ratings.groupby('user')['rating'].count().mean()
avg_ratings_per_course = ratings.groupby('item')['rating'].count().mean()
avg_courses_per_user = user_enrollments['courses_enrolled'].mean()
median_courses_per_user = user_enrollments['courses_enrolled'].median()
sparsity = 1 - (len(ratings) / (ratings['user'].nunique() * ratings['item'].nunique()))

insights = f"""
- **Data Sparsity**: {sparsity:.2%} - The rating matrix is highly sparse, which is typical for recommendation systems
- **Average ratings per user**: {avg_ratings_per_user:.1f}
- **Average ratings per course**: {avg_ratings_per_course:.1f}
- **Average courses per user**: {avg_courses_per_user:.1f} (Median: {median_courses_per_user:.0f})
- **Rating Skew**: Most ratings tend toward {ratings['rating'].mode()[0]} (mode), indicating {"positive" if ratings['rating'].mode()[0] > 3 else "negative"} bias
- **User Engagement**: {(len(user_activity[user_activity['ratings_count'] >= 5]) / len(user_activity) * 100):.1f}% of users have rated 5 or more courses
- **Single Course Users**: {(enrollment_dist[enrollment_dist['num_courses']==1]['num_users'].values[0] / total_users * 100):.1f}% of users enrolled in only one course
"""

st.info(insights)
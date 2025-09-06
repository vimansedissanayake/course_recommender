# pages/8_Evaluation.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Algorithm Evaluation", layout="wide")
st.title("🏆 Collaborative Filtering Algorithms Evaluation")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    courses = pd.read_csv("data/course_genre.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return courses, ratings

courses, ratings = load_data()

# Evaluation metrics functions
def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def precision_at_k(actual, predicted, k=10):
    """Calculate Precision@K"""
    if len(predicted) > k:
        predicted = predicted[:k]
    
    if not actual or not predicted:
        return 0.0
    
    return len(set(actual) & set(predicted)) / len(predicted)

def recall_at_k(actual, predicted, k=10):
    """Calculate Recall@K"""
    if len(predicted) > k:
        predicted = predicted[:k]
    
    if not actual:
        return 0.0
    
    return len(set(actual) & set(predicted)) / len(actual)

def f1_at_k(actual, predicted, k=10):
    """Calculate F1@K"""
    prec = precision_at_k(actual, predicted, k)
    rec = recall_at_k(actual, predicted, k)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)

def ndcg_at_k(actual, predicted, k=10):
    """Calculate NDCG@K (Normalized Discounted Cumulative Gain)"""
    if len(predicted) > k:
        predicted = predicted[:k]
    
    if not actual or not predicted:
        return 0.0
    
    dcg = 0.0
    for i, item in enumerate(predicted):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because index starts at 0
    
    # Ideal DCG
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual), k))])
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def coverage(recommendations, all_items):
    """Calculate catalog coverage"""
    recommended_items = set()
    for items in recommendations.values():
        recommended_items.update(items)
    
    return len(recommended_items) / len(all_items)

# Evaluation functions for each algorithm
@st.cache_data
def evaluate_knn(ratings_df, k_neighbors=5, test_size=0.2):
    """Evaluate KNN-based collaborative filtering"""
    start_time = time.time()
    
    # Split data
    train_data, test_data = train_test_split(ratings_df, test_size=test_size, random_state=42)
    
    # Create user-item matrix
    user_item_train = train_data.pivot_table(
        index='user', columns='item', values='rating', fill_value=0
    )
    user_item_test = test_data.pivot_table(
        index='user', columns='item', values='rating', fill_value=0
    )
    
    # Train KNN model
    user_item_sparse = csr_matrix(user_item_train.values)
    knn_model = NearestNeighbors(n_neighbors=k_neighbors+1, metric='cosine', algorithm='brute')
    knn_model.fit(user_item_sparse)
    
    # Make predictions for test set
    predictions = []
    actuals = []
    
    # Sample for efficiency
    sample_users = np.random.choice(
        test_data['user'].unique(), 
        min(100, len(test_data['user'].unique())), 
        replace=False
    )
    
    for user in sample_users:
        if user in user_item_train.index:
            user_idx = user_item_train.index.get_loc(user)
            distances, indices = knn_model.kneighbors(
                user_item_train.iloc[user_idx].values.reshape(1, -1),
                n_neighbors=k_neighbors+1
            )
            
            # Get test items for this user
            user_test_data = test_data[test_data['user'] == user]
            
            for _, row in user_test_data.iterrows():
                item = row['item']
                if item in user_item_train.columns:
                    item_idx = user_item_train.columns.get_loc(item)
                    
                    # Predict rating as weighted average of neighbors
                    neighbor_ratings = []
                    weights = []
                    
                    for idx, dist in zip(indices.flatten()[1:], distances.flatten()[1:]):
                        rating = user_item_train.iloc[idx, item_idx]
                        if rating > 0:
                            neighbor_ratings.append(rating)
                            weights.append(1 - dist)
                    
                    if neighbor_ratings:
                        pred = np.average(neighbor_ratings, weights=weights)
                        predictions.append(pred)
                        actuals.append(row['rating'])
    
    # Calculate metrics
    if predictions:
        rmse = calculate_rmse(actuals, predictions)
        mae = calculate_mae(actuals, predictions)
    else:
        rmse = mae = float('inf')
    
    training_time = time.time() - start_time
    
    # Calculate ranking metrics (simplified)
    precision = recall = f1 = ndcg = 0.0
    sample_size = min(20, len(sample_users))
    
    for user in sample_users[:sample_size]:
        if user in user_item_train.index:
            # Get actual high-rated items
            user_test = test_data[test_data['user'] == user]
            actual_items = user_test[user_test['rating'] >= 4]['item'].tolist()
            
            if actual_items:
                # Get predicted top items (simplified)
                user_idx = user_item_train.index.get_loc(user)
                distances, indices = knn_model.kneighbors(
                    user_item_train.iloc[user_idx].values.reshape(1, -1),
                    n_neighbors=min(20, len(user_item_train))
                )
                
                predicted_items = []
                for idx in indices.flatten()[1:11]:  # Top 10
                    # Get items this neighbor rated highly
                    neighbor_items = user_item_train.columns[user_item_train.iloc[idx] >= 4].tolist()
                    predicted_items.extend(neighbor_items[:2])
                
                predicted_items = list(set(predicted_items))[:10]
                
                precision += precision_at_k(actual_items, predicted_items)
                recall += recall_at_k(actual_items, predicted_items)
                f1 += f1_at_k(actual_items, predicted_items)
                ndcg += ndcg_at_k(actual_items, predicted_items)
    
    if sample_size > 0:
        precision /= sample_size
        recall /= sample_size
        f1 /= sample_size
        ndcg /= sample_size
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': precision,
        'Recall@10': recall,
        'F1@10': f1,
        'NDCG@10': ndcg,
        'Training Time (s)': training_time
    }

@st.cache_data
def evaluate_nmf(ratings_df, n_components=20, test_size=0.2):
    """Evaluate NMF-based collaborative filtering"""
    start_time = time.time()
    
    # Split data
    train_data, test_data = train_test_split(ratings_df, test_size=test_size, random_state=42)
    
    # Create user-item matrix
    user_item_train = train_data.pivot_table(
        index='user', columns='item', values='rating', fill_value=0
    )
    
    # Prepare data for NMF
    matrix_filled = user_item_train.copy()
    matrix_filled[matrix_filled == 0] = 0.1
    
    # Train NMF model
    nmf_model = NMF(n_components=n_components, random_state=42, max_iter=200)
    W = nmf_model.fit_transform(matrix_filled)
    H = nmf_model.components_
    
    # Make predictions for test set
    predictions = []
    actuals = []
    
    for _, row in test_data.iterrows():
        user = row['user']
        item = row['item']
        
        if user in user_item_train.index and item in user_item_train.columns:
            user_idx = user_item_train.index.get_loc(user)
            item_idx = user_item_train.columns.get_loc(item)
            
            pred = W[user_idx] @ H[:, item_idx]
            predictions.append(pred)
            actuals.append(row['rating'])
    
    # Calculate metrics
    if predictions:
        rmse = calculate_rmse(actuals, predictions)
        mae = calculate_mae(actuals, predictions)
    else:
        rmse = mae = float('inf')
    
    training_time = time.time() - start_time
    
    # Calculate ranking metrics
    precision = recall = f1 = ndcg = 0.0
    sample_users = test_data['user'].unique()[:20]
    
    for user in sample_users:
        if user in user_item_train.index:
            user_idx = user_item_train.index.get_loc(user)
            
            # Get actual high-rated items
            user_test = test_data[test_data['user'] == user]
            actual_items = user_test[user_test['rating'] >= 4]['item'].tolist()
            
            if actual_items:
                # Predict ratings for all items
                user_predictions = W[user_idx] @ H
                
                # Get top 10 predicted items (excluding already rated)
                user_train = train_data[train_data['user'] == user]['item'].tolist()
                
                item_scores = []
                for item_idx, score in enumerate(user_predictions):
                    item = user_item_train.columns[item_idx]
                    if item not in user_train:
                        item_scores.append((item, score))
                
                item_scores.sort(key=lambda x: x[1], reverse=True)
                predicted_items = [item for item, _ in item_scores[:10]]
                
                precision += precision_at_k(actual_items, predicted_items)
                recall += recall_at_k(actual_items, predicted_items)
                f1 += f1_at_k(actual_items, predicted_items)
                ndcg += ndcg_at_k(actual_items, predicted_items)
    
    if len(sample_users) > 0:
        precision /= len(sample_users)
        recall /= len(sample_users)
        f1 /= len(sample_users)
        ndcg /= len(sample_users)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': precision,
        'Recall@10': recall,
        'F1@10': f1,
        'NDCG@10': ndcg,
        'Training Time (s)': training_time
    }

@st.cache_data
def evaluate_simple_baseline(ratings_df, test_size=0.2):
    """Evaluate simple baseline (global mean or popularity-based)"""
    start_time = time.time()
    
    # Split data
    train_data, test_data = train_test_split(ratings_df, test_size=test_size, random_state=42)
    
    # Calculate global mean rating
    global_mean = train_data['rating'].mean()
    
    # Calculate item popularity (average rating)
    item_popularity = train_data.groupby('item')['rating'].agg(['mean', 'count'])
    item_popularity = item_popularity[item_popularity['count'] >= 5]  # Min 5 ratings
    item_popularity = item_popularity.sort_values('mean', ascending=False)
    
    # Predictions using global mean
    predictions = [global_mean] * len(test_data)
    actuals = test_data['rating'].tolist()
    
    # Calculate error metrics
    rmse = calculate_rmse(actuals, predictions)
    mae = calculate_mae(actuals, predictions)
    
    training_time = time.time() - start_time
    
    # Calculate ranking metrics (recommend most popular items)
    precision = recall = f1 = ndcg = 0.0
    sample_users = test_data['user'].unique()[:20]
    
    top_popular_items = item_popularity.head(10).index.tolist()
    
    for user in sample_users:
        # Get actual high-rated items
        user_test = test_data[test_data['user'] == user]
        actual_items = user_test[user_test['rating'] >= 4]['item'].tolist()
        
        if actual_items:
            precision += precision_at_k(actual_items, top_popular_items)
            recall += recall_at_k(actual_items, top_popular_items)
            f1 += f1_at_k(actual_items, top_popular_items)
            ndcg += ndcg_at_k(actual_items, top_popular_items)
    
    if len(sample_users) > 0:
        precision /= len(sample_users)
        recall /= len(sample_users)
        f1 /= len(sample_users)
        ndcg /= len(sample_users)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': precision,
        'Recall@10': recall,
        'F1@10': f1,
        'NDCG@10': ndcg,
        'Training Time (s)': training_time
    }

# Main UI
st.markdown("### Comprehensive Algorithm Comparison")
st.info("""
This page evaluates and compares different collaborative filtering algorithms on your dataset:
- **Baseline**: Global mean and popularity-based recommendations
- **KNN**: K-Nearest Neighbors collaborative filtering
- **NMF**: Non-negative Matrix Factorization
- **Neural Network**: Deep learning-based (if trained)
""")

# Evaluation settings
with st.sidebar:
    st.header("⚙️ Evaluation Settings")
    
    test_size = st.slider(
        "Test Set Size",
        min_value=0.1,
        max_value=0.3,
        value=0.2,
        step=0.05,
        help="Proportion of data to use for testing"
    )
    
    k_neighbors = st.slider(
        "KNN Neighbors",
        min_value=3,
        max_value=20,
        value=5,
        help="Number of neighbors for KNN"
    )
    
    n_components = st.slider(
        "NMF Components",
        min_value=5,
        max_value=50,
        value=20,
        help="Number of latent factors for NMF"
    )
    
    st.markdown("---")
    st.subheader("📊 Dataset Statistics")
    st.metric("Total Ratings", len(ratings))
    st.metric("Unique Users", ratings['user'].nunique())
    st.metric("Unique Items", ratings['item'].nunique())
    sparsity = 1 - len(ratings) / (ratings['user'].nunique() * ratings['item'].nunique())
    st.metric("Sparsity", f"{sparsity:.2%}")

# Run evaluation
if st.button("🚀 Run Evaluation", type="primary"):
    with st.spinner("Evaluating algorithms... This may take a few minutes."):
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        
        # Evaluate Baseline
        status_text.text("Evaluating Baseline...")
        progress_bar.progress(0.25)
        results['Baseline'] = evaluate_simple_baseline(ratings, test_size)
        
        # Evaluate KNN
        status_text.text("Evaluating KNN...")
        progress_bar.progress(0.50)
        results['KNN'] = evaluate_knn(ratings, k_neighbors, test_size)
        
        # Evaluate NMF
        status_text.text("Evaluating NMF...")
        progress_bar.progress(0.75)
        results['NMF'] = evaluate_nmf(ratings, n_components, test_size)
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("Evaluation complete!")
        
        # Store results
        st.session_state['evaluation_results'] = results
        st.success("✅ Evaluation completed successfully!")

# Display results
if 'evaluation_results' in st.session_state:
    results = st.session_state['evaluation_results']
    
    # Create results dataframe
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Metrics Overview", 
        "📈 Visual Comparison", 
        "🎯 Detailed Analysis",
        "💡 Recommendations"
    ])
    
    with tab1:
        st.subheader("📊 Evaluation Metrics")
        
        # Display main metrics table
        st.dataframe(
            results_df.style.highlight_min(axis=0, subset=['RMSE', 'MAE', 'Training Time (s)'], color='lightgreen')
                           .highlight_max(axis=0, subset=['Precision@10', 'Recall@10', 'F1@10', 'NDCG@10'], color='lightgreen'),
            use_container_width=True
        )
        
        # Best performers
        st.markdown("#### 🏆 Best Performers")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_rmse = results_df['RMSE'].idxmin()
            st.metric("Best RMSE", best_rmse, f"{results_df.loc[best_rmse, 'RMSE']:.4f}")
        
        with col2:
            best_precision = results_df['Precision@10'].idxmax()
            st.metric("Best Precision@10", best_precision, f"{results_df.loc[best_precision, 'Precision@10']:.4f}")
        
        with col3:
            fastest = results_df['Training Time (s)'].idxmin()
            st.metric("Fastest Training", fastest, f"{results_df.loc[fastest, 'Training Time (s)']:.2f}s")
    
    with tab2:
        st.subheader("📈 Visual Comparison")
        
        # Error metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                results_df[['RMSE', 'MAE']].T,
                title="Error Metrics Comparison",
                labels={'index': 'Metric', 'value': 'Score'},
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                results_df[['Precision@10', 'Recall@10', 'F1@10', 'NDCG@10']].T,
                title="Ranking Metrics Comparison",
                labels={'index': 'Metric', 'value': 'Score'},
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart for comprehensive comparison
        st.markdown("#### 🎯 Comprehensive Performance Radar")
        
        # Normalize metrics for radar chart (0-1 scale)
        normalized_df = results_df.copy()
        
        # Invert error metrics (lower is better)
        for col in ['RMSE', 'MAE', 'Training Time (s)']:
            if col in normalized_df.columns:
                max_val = normalized_df[col].max()
                if max_val > 0:
                    normalized_df[col] = 1 - (normalized_df[col] / max_val)
        
        # Normalize ranking metrics (already 0-1)
        
        fig = go.Figure()
        
        for algorithm in normalized_df.index:
            fig.add_trace(go.Scatterpolar(
                r=normalized_df.loc[algorithm].values,
                theta=normalized_df.columns,
                fill='toself',
                name=algorithm
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Algorithm Performance Radar (Normalized)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Time vs Performance scatter
        st.markdown("#### ⚡ Efficiency vs Accuracy")
        
        fig = px.scatter(
            results_df.reset_index(),
            x='Training Time (s)',
            y='RMSE',
            size='F1@10',
            color='index',
            title="Training Time vs RMSE (bubble size = F1@10)",
            labels={'index': 'Algorithm'},
            hover_data=['MAE', 'Precision@10', 'Recall@10'],
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Detailed Analysis")
        
        # Statistical significance (simplified)
        st.markdown("#### 📊 Performance Summary")
        
        # Create summary statistics
        summary = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'Precision@10', 'Recall@10', 'F1@10', 'NDCG@10'],
            'Best Algorithm': [
                results_df['RMSE'].idxmin(),
                results_df['MAE'].idxmin(),
                results_df['Precision@10'].idxmax(),
                results_df['Recall@10'].idxmax(),
                results_df['F1@10'].idxmax(),
                results_df['NDCG@10'].idxmax()
            ],
            'Best Score': [
                results_df['RMSE'].min(),
                results_df['MAE'].min(),
                results_df['Precision@10'].max(),
                results_df['Recall@10'].max(),
                results_df['F1@10'].max(),
                results_df['NDCG@10'].max()
            ],
            'Improvement over Baseline': [
                f"{(results_df.loc['Baseline', 'RMSE'] - results_df['RMSE'].min()) / results_df.loc['Baseline', 'RMSE'] * 100:.1f}%",
                f"{(results_df.loc['Baseline', 'MAE'] - results_df['MAE'].min()) / results_df.loc['Baseline', 'MAE'] * 100:.1f}%",
                f"{(results_df['Precision@10'].max() - results_df.loc['Baseline', 'Precision@10']) / max(results_df.loc['Baseline', 'Precision@10'], 0.001) * 100:.1f}%",
                f"{(results_df['Recall@10'].max() - results_df.loc['Baseline', 'Recall@10']) / max(results_df.loc['Baseline', 'Recall@10'], 0.001) * 100:.1f}%",
                f"{(results_df['F1@10'].max() - results_df.loc['Baseline', 'F1@10']) / max(results_df.loc['Baseline', 'F1@10'], 0.001) * 100:.1f}%",
                f"{(results_df['NDCG@10'].max() - results_df.loc['Baseline', 'NDCG@10']) / max(results_df.loc['Baseline', 'NDCG@10'], 0.001) * 100:.1f}%"
            ]
        })
        
        st.dataframe(summary, use_container_width=True, hide_index=True)
        
        # Trade-offs analysis
        st.markdown("#### ⚖️ Trade-offs Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Accuracy vs Speed**")
            accuracy_speed = results_df[['RMSE', 'Training Time (s)']].copy()
            accuracy_speed['Accuracy Score'] = 1 / accuracy_speed['RMSE']
            accuracy_speed['Speed Score'] = 1 / accuracy_speed['Training Time (s)']
            
            fig = px.scatter(
                accuracy_speed.reset_index(),
                x='Speed Score',
                y='Accuracy Score',
                text='index',
                title="Accuracy vs Speed Trade-off"
            )
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Precision vs Recall**")
            
            fig = px.scatter(
                results_df.reset_index(),
                x='Recall@10',
                y='Precision@10',
                text='index',
                title="Precision-Recall Trade-off"
            )
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("💡 Recommendations")
        
        # Analyze results and provide recommendations
        best_overall = results_df['F1@10'].idxmax()
        best_accuracy = results_df['RMSE'].idxmin()
        fastest = results_df['Training Time (s)'].idxmin()
        
        st.markdown("### 🎯 Algorithm Selection Guide")
        
        st.markdown(f"""
        Based on the evaluation results, here are our recommendations:
        
        #### **Best Overall: {best_overall}**
        - Balanced performance across all metrics
        - F1@10 Score: {results_df.loc[best_overall, 'F1@10']:.4f}
        - Good for general-purpose recommendations
        
        #### **Most Accurate: {best_accuracy}**
        - Lowest prediction error (RMSE: {results_df.loc[best_accuracy, 'RMSE']:.4f})
        - Best for rating prediction tasks
        - Suitable when accuracy is paramount
        
        #### **Fastest: {fastest}**
        - Training time: {results_df.loc[fastest, 'Training Time (s)']:.2f} seconds
        - Good for real-time applications
        - Suitable for large-scale deployments
        """)
        
        # Contextual recommendations
        st.markdown("### 📋 Use Case Recommendations")
        
        use_cases = {
            "Small Dataset (<10K ratings)": "KNN - Simple and effective",
            "Medium Dataset (10K-100K ratings)": "NMF - Good balance",
            "Large Dataset (>100K ratings)": "Neural Network - Scales well",
            "Real-time Requirements": fastest,
            "High Accuracy Needed": best_accuracy,
            "Interpretability Important": "NMF - Interpretable factors",
            "Cold Start Handling": "Content-based or Hybrid approaches"
        }
        
        use_case_df = pd.DataFrame(
            list(use_cases.items()),
            columns=['Use Case', 'Recommended Algorithm']
        )
        
        st.dataframe(use_case_df, use_container_width=True, hide_index=True)
        
        # Dataset-specific insights
        st.markdown("### 📊 Dataset-Specific Insights")
        
        if sparsity > 0.99:
            st.warning("""
            **High Sparsity Detected** ({:.2%})
            - Consider using matrix factorization methods (NMF)
            - Hybrid approaches may help
            - More data collection recommended
            """.format(sparsity))
        elif sparsity > 0.95:
            st.info("""
            **Moderate Sparsity** ({:.2%})
            - Current algorithms should work well
            - Consider dimensionality reduction
            - Ensemble methods may improve results
            """.format(sparsity))
        else:
            st.success("""
            **Good Data Density** ({:.2%})
            - All algorithms should perform well
            - Deep learning methods recommended
            - Consider more complex models
            """.format(sparsity))

# Export results
if 'evaluation_results' in st.session_state:
    st.markdown("---")
    st.markdown("### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as CSV
        csv = results_df.to_csv()
        st.download_button(
            label="Download Results (CSV)",
            data=csv,
            file_name="evaluation_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export as JSON
        import json
        json_str = results_df.to_json(orient='index', indent=2)
        st.download_button(
            label="Download Results (JSON)",
            data=json_str,
            file_name="evaluation_results.json",
            mime="application/json"
        )
    
    with col3:
        # Export summary report
        report = f"""
        Collaborative Filtering Evaluation Report
        =========================================
        
        Dataset Statistics:
        - Total Ratings: {len(ratings)}
        - Unique Users: {ratings['user'].nunique()}
        - Unique Items: {ratings['item'].nunique()}
        - Sparsity: {sparsity:.2%}
        
        Best Performers:
        - Overall: {best_overall}
        - Accuracy: {best_accuracy}
        - Speed: {fastest}
        
        Detailed Results:
        {results_df.to_string()}
        
        Generated: {pd.Timestamp.now()}
        """
        
        st.download_button(
            label="Download Report (TXT)",
            data=report,
            file_name="evaluation_report.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("### 📚 Metrics Explained")
with st.expander("Understanding Evaluation Metrics"):
    st.markdown("""
    **Error Metrics:**
    - **RMSE** (Root Mean Squared Error): Measures prediction accuracy, lower is better
    - **MAE** (Mean Absolute Error): Average prediction error, lower is better
    
    **Ranking Metrics:**
    - **Precision@K**: Fraction of recommended items that are relevant
    - **Recall@K**: Fraction of relevant items that are recommended
    - **F1@K**: Harmonic mean of Precision and Recall
    - **NDCG@K**: Measures ranking quality with position-based weights
    
    **Efficiency Metrics:**
    - **Training Time**: Time to train the model in seconds
    - **Prediction Time**: Time to generate recommendations (not shown)
    
    **Interpretation:**
    - Error metrics measure how well the model predicts exact ratings
    - Ranking metrics measure how well the model identifies relevant items
    - Trade-offs exist between accuracy, coverage, and speed
    """)
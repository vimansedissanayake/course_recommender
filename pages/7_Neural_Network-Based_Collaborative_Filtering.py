# pages/7_Neural_Network-Based_Collaborative_Filtering.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

st.set_page_config(page_title="Neural Collaborative Filtering", layout="wide")
st.title("🧠 Neural Network Embedding-Based Collaborative Filtering")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    courses = pd.read_csv("data/course_genre.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return courses, ratings

courses, ratings = load_data()

# Preprocess data for neural network
@st.cache_data
def preprocess_data(ratings_df):
    """Preprocess data for neural network training"""
    
    # Create copies
    data = ratings_df.copy()
    
    # Encode user and item IDs to sequential integers
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    data['user_encoded'] = user_encoder.fit_transform(data['user'])
    data['item_encoded'] = item_encoder.fit_transform(data['item'])
    
    # Get number of unique users and items
    n_users = data['user_encoded'].nunique()
    n_items = data['item_encoded'].nunique()
    
    # Normalize ratings to [0, 1] for better training
    min_rating = data['rating'].min()
    max_rating = data['rating'].max()
    data['rating_normalized'] = (data['rating'] - min_rating) / (max_rating - min_rating)
    
    return data, user_encoder, item_encoder, n_users, n_items, min_rating, max_rating

# Build neural network model
def build_embedding_model(n_users, n_items, embedding_dim=50, architecture='dot'):
    """Build neural collaborative filtering model"""
    
    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    
    # Embedding layers
    user_embedding = Embedding(n_users, embedding_dim, name='user_embedding')(user_input)
    item_embedding = Embedding(n_items, embedding_dim, name='item_embedding')(item_input)
    
    # Flatten embeddings
    user_vec = Flatten(name='user_flatten')(user_embedding)
    item_vec = Flatten(name='item_flatten')(item_embedding)
    
    if architecture == 'dot':
        # Simple dot product model
        dot_product = Dot(axes=1, name='dot_product')([user_vec, item_vec])
        output = Dense(1, activation='sigmoid', name='output')(dot_product)
        
    elif architecture == 'mlp':
        # Multi-layer perceptron model
        concat = Concatenate()([user_vec, item_vec])
        dense1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation='relu')(dropout2)
        output = Dense(1, activation='sigmoid', name='output')(dense3)
        
    else:  # 'hybrid'
        # Hybrid model combining dot product and MLP
        dot_product = Dot(axes=1)([user_vec, item_vec])
        concat = Concatenate()([user_vec, item_vec, dot_product])
        dense1 = Dense(64, activation='relu')(concat)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        output = Dense(1, activation='sigmoid', name='output')(dense2)
    
    model = Model(inputs=[user_input, item_input], outputs=output)
    return model

# Train model function
@st.cache_resource
def train_model(data, n_users, n_items, embedding_dim, architecture, epochs, batch_size):
    """Train the neural collaborative filtering model"""
    
    # Split data
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42
    )
    
    # Prepare training data
    X_train = [train_data['user_encoded'].values, train_data['item_encoded'].values]
    y_train = train_data['rating_normalized'].values
    
    X_test = [test_data['user_encoded'].values, test_data['item_encoded'].values]
    y_test = test_data['rating_normalized'].values
    
    # Build and compile model
    model = build_embedding_model(n_users, n_items, embedding_dim, architecture)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    return model, history, test_loss, test_mae

# Generate recommendations
def get_nn_recommendations(user_id, model, data, user_encoder, item_encoder, 
                          min_rating, max_rating, courses_df, n_recommendations=10):
    """Generate recommendations using the trained neural network"""
    
    try:
        # Encode user ID
        user_encoded = user_encoder.transform([user_id])[0]
    except:
        return pd.DataFrame(), "User not found"
    
    # Get all items
    all_items = data['item'].unique()
    
    # Get items already rated by user
    user_rated = data[data['user'] == user_id]['item'].tolist()
    
    # Get items not rated by user
    items_to_predict = [item for item in all_items if item not in user_rated]
    
    if not items_to_predict:
        return pd.DataFrame(), "User has rated all items"
    
    # Encode items
    items_encoded = item_encoder.transform(items_to_predict)
    
    # Create prediction input
    user_array = np.array([user_encoded] * len(items_encoded))
    items_array = np.array(items_encoded)
    
    # Make predictions
    predictions_normalized = model.predict([user_array, items_array], verbose=0).flatten()
    
    # Denormalize predictions
    predictions = predictions_normalized * (max_rating - min_rating) + min_rating
    
    # Create recommendations dataframe
    recommendations = pd.DataFrame({
        'item': items_to_predict,
        'predicted_rating': predictions
    })
    
    # Get top recommendations
    recommendations = recommendations.nlargest(n_recommendations, 'predicted_rating')
    
    # Add course information
    recommendations = recommendations.merge(
        courses_df[['COURSE_ID', 'TITLE']], 
        left_on='item', 
        right_on='COURSE_ID'
    )
    
    # Add actual statistics
    course_stats = data.groupby('item')['rating'].agg(['mean', 'count', 'std']).reset_index()
    recommendations = recommendations.merge(course_stats, on='item', how='left')
    
    return recommendations, None

# Get user embeddings for visualization
def get_user_embeddings(model, n_users):
    """Extract user embeddings from the model"""
    user_embedding_layer = model.get_layer('user_embedding')
    user_embeddings = user_embedding_layer.get_weights()[0]
    return user_embeddings

# Get item embeddings for visualization
def get_item_embeddings(model, n_items):
    """Extract item embeddings from the model"""
    item_embedding_layer = model.get_layer('item_embedding')
    item_embeddings = item_embedding_layer.get_weights()[0]
    return item_embeddings

# Streamlit UI
st.markdown("### How it works")
st.info("""
Neural Collaborative Filtering uses deep learning to learn user and item embeddings:
- **Embeddings**: Dense vector representations of users and items
- **Architecture Options**: Dot product, MLP, or Hybrid models
- **Training**: Learns patterns from user-item interactions
- **Predictions**: Uses learned embeddings to predict ratings
""")

# Preprocess data
with st.spinner("Preprocessing data..."):
    data, user_encoder, item_encoder, n_users, n_items, min_rating, max_rating = preprocess_data(ratings)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    embedding_dim = st.slider(
        "Embedding Dimension",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="Size of embedding vectors"
    )
    
    architecture = st.selectbox(
        "Model Architecture",
        options=['dot', 'mlp', 'hybrid'],
        format_func=lambda x: {
            'dot': 'Dot Product',
            'mlp': 'Multi-Layer Perceptron',
            'hybrid': 'Hybrid (Dot + MLP)'
        }[x],
        help="Choose the neural network architecture"
    )
    
    epochs = st.slider(
        "Training Epochs",
        min_value=5,
        max_value=50,
        value=20,
        help="Number of training epochs"
    )
    
    batch_size = st.slider(
        "Batch Size",
        min_value=32,
        max_value=512,
        value=128,
        step=32,
        help="Training batch size"
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
    st.metric("Total Users", n_users)
    st.metric("Total Items", n_items)
    st.metric("Total Ratings", len(data))
    st.metric("Embedding Size", f"{n_users}×{embedding_dim} + {n_items}×{embedding_dim}")

# Train model
if st.button("🚀 Train Model", type="primary"):
    with st.spinner(f"Training {architecture} model for {epochs} epochs..."):
        model, history, test_loss, test_mae = train_model(
            data, n_users, n_items, embedding_dim, architecture, epochs, batch_size
        )
        st.session_state['model'] = model
        st.session_state['history'] = history
        st.session_state['test_loss'] = test_loss
        st.session_state['test_mae'] = test_mae
        st.success(f"✅ Model trained! Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Main interface tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Get Recommendations", 
    "📈 Training History", 
    "🧬 Embeddings Visualization",
    "🔍 Model Analysis",
    "📊 Prediction Analysis"
])

with tab1:
    if 'model' in st.session_state:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📝 Select User")
            
            # User selection
            search_user = st.text_input("Search User ID:", "")
            
            if search_user:
                filtered_users = [u for u in data['user'].unique() if str(search_user) in str(u)]
            else:
                filtered_users = list(data['user'].unique()[:100])  # Convert to list and limit for performance
            
            if len(filtered_users) > 0:  # Fix: Use len() instead of direct truthiness
                user_id = st.selectbox(
                    "Select a User ID",
                    options=filtered_users,
                    help="Choose a user to generate recommendations"
                )
                
                # Show user info
                if user_id:
                    user_data = data[data['user'] == user_id]
                    
                    st.markdown("#### 👤 User Statistics")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Total Ratings", len(user_data))
                    with col_b:
                        st.metric("Avg Rating", f"{user_data['rating'].mean():.2f}")
                    
                    # Generate recommendations
                    if st.button("🎯 Get Recommendations"):
                        with st.spinner("Generating recommendations..."):
                            recommendations, error = get_nn_recommendations(
                                user_id, 
                                st.session_state['model'],
                                data,
                                user_encoder,
                                item_encoder,
                                min_rating,
                                max_rating,
                                courses,
                                n_recommendations
                            )
                            
                            st.session_state['recommendations'] = recommendations
                            st.session_state['rec_error'] = error
                            st.session_state['selected_user'] = user_id
        
        with col2:
            st.subheader("📚 Recommendations")
            
            if 'recommendations' in st.session_state:
                if st.session_state.get('rec_error'):
                    st.error(st.session_state['rec_error'])
                elif not st.session_state['recommendations'].empty:
                    st.success(f"✅ Generated {len(st.session_state['recommendations'])} recommendations")
                    
                    # Display recommendations
                    display_df = st.session_state['recommendations'][
                        ['TITLE', 'predicted_rating', 'mean', 'count']
                    ].copy()
                    display_df.columns = ['Course Title', 'Predicted Rating', 'Actual Avg Rating', 'Total Ratings']
                    display_df['Predicted Rating'] = display_df['Predicted Rating'].round(2)
                    display_df['Actual Avg Rating'] = display_df['Actual Avg Rating'].fillna(0).round(2)
                    display_df['Total Ratings'] = display_df['Total Ratings'].fillna(0).astype(int)
                    display_df.index = range(1, len(display_df) + 1)
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Visualization
                    fig = px.scatter(
                        st.session_state['recommendations'].dropna(),
                        x='mean',
                        y='predicted_rating',
                        size='count',
                        hover_data=['TITLE'],
                        title='Neural Network Predictions vs Actual Ratings',
                        labels={
                            'mean': 'Actual Average Rating',
                            'predicted_rating': 'NN Predicted Rating',
                            'count': 'Number of Ratings'
                        },
                        color='predicted_rating',
                        color_continuous_scale='RdYlGn'
                    )
                    
                    # Add diagonal line
                    fig.add_trace(go.Scatter(
                        x=[1, 5],
                        y=[1, 5],
                        mode='lines',
                        line=dict(dash='dash', color='gray'),
                        showlegend=False
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please train the model first using the button in the sidebar")

with tab2:
    if 'history' in st.session_state:
        st.subheader("📈 Training History")
        
        history_df = pd.DataFrame(st.session_state['history'].history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df.index + 1,
                y=history_df['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=history_df.index + 1,
                y=history_df['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red')
            ))
            fig.update_layout(
                title='Model Loss',
                xaxis_title='Epoch',
                yaxis_title='Loss (MSE)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # MAE plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df.index + 1,
                y=history_df['mae'],
                mode='lines',
                name='Training MAE',
                line=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=history_df.index + 1,
                y=history_df['val_mae'],
                mode='lines',
                name='Validation MAE',
                line=dict(color='orange')
            ))
            fig.update_layout(
                title='Model MAE',
                xaxis_title='Epoch',
                yaxis_title='MAE',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Final metrics
        st.markdown("#### 📊 Final Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Train Loss", f"{history_df['loss'].iloc[-1]:.4f}")
        with col2:
            st.metric("Final Val Loss", f"{history_df['val_loss'].iloc[-1]:.4f}")
        with col3:
            st.metric("Test Loss", f"{st.session_state.get('test_loss', 0):.4f}")
        with col4:
            st.metric("Test MAE", f"{st.session_state.get('test_mae', 0):.4f}")
    else:
        st.info("Train the model to see training history")

with tab3:
    if 'model' in st.session_state:
        st.subheader("🧬 Embeddings Visualization")
        
        # Get embeddings
        user_embeddings = get_user_embeddings(st.session_state['model'], n_users)
        item_embeddings = get_item_embeddings(st.session_state['model'], n_items)
        
        # PCA for visualization
        from sklearn.decomposition import PCA
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👥 User Embeddings (PCA)")
            
            if user_embeddings.shape[1] >= 2:
                pca_user = PCA(n_components=2)
                user_embeddings_2d = pca_user.fit_transform(user_embeddings[:min(500, len(user_embeddings))])
                
                fig = px.scatter(
                    x=user_embeddings_2d[:, 0],
                    y=user_embeddings_2d[:, 1],
                    title="User Embeddings in 2D Space",
                    labels={'x': 'First Component', 'y': 'Second Component'},
                    color=np.arange(len(user_embeddings_2d)),
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"PCA explained variance: {pca_user.explained_variance_ratio_.sum():.2%}")
        
        with col2:
            st.markdown("#### 📚 Item Embeddings (PCA)")
            
            if item_embeddings.shape[1] >= 2:
                pca_item = PCA(n_components=2)
                item_embeddings_2d = pca_item.fit_transform(item_embeddings[:min(500, len(item_embeddings))])
                
                fig = px.scatter(
                    x=item_embeddings_2d[:, 0],
                    y=item_embeddings_2d[:, 1],
                    title="Item Embeddings in 2D Space",
                    labels={'x': 'First Component', 'y': 'Second Component'},
                    color=np.arange(len(item_embeddings_2d)),
                    color_continuous_scale='Plasma'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"PCA explained variance: {pca_item.explained_variance_ratio_.sum():.2%}")
        
        # Embedding statistics
        st.markdown("#### 📊 Embedding Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**User Embeddings**")
            st.metric("Shape", f"{user_embeddings.shape}")
            st.metric("Mean Norm", f"{np.mean(np.linalg.norm(user_embeddings, axis=1)):.3f}")
            st.metric("Std Norm", f"{np.std(np.linalg.norm(user_embeddings, axis=1)):.3f}")
        
        with col2:
            st.markdown("**Item Embeddings**")
            st.metric("Shape", f"{item_embeddings.shape}")
            st.metric("Mean Norm", f"{np.mean(np.linalg.norm(item_embeddings, axis=1)):.3f}")
            st.metric("Std Norm", f"{np.std(np.linalg.norm(item_embeddings, axis=1)):.3f}")
    else:
        st.info("Train the model to visualize embeddings")

with tab4:
    if 'model' in st.session_state:
        st.subheader("🔍 Model Analysis")
        
        # Model summary
        st.markdown("#### 🏗️ Model Architecture")
        
        model_summary = []
        st.session_state['model'].summary(print_fn=lambda x: model_summary.append(x))
        
        st.text('\n'.join(model_summary))
        
        # Parameter count
        total_params = st.session_state['model'].count_params()
        st.metric("Total Parameters", f"{total_params:,}")
        
        # Sample predictions
        st.markdown("#### 🎲 Sample Predictions")
        
        sample_size = min(100, len(data))
        sample_data = data.sample(sample_size)
        
        X_sample = [sample_data['user_encoded'].values, sample_data['item_encoded'].values]
        y_true = sample_data['rating'].values
        
        y_pred_normalized = st.session_state['model'].predict(X_sample, verbose=0).flatten()
        y_pred = y_pred_normalized * (max_rating - min_rating) + min_rating
        
        # Error analysis
        errors = np.abs(y_true - y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                x=y_true,
                y=y_pred,
                title="True vs Predicted Ratings (Sample)",
                labels={'x': 'True Rating', 'y': 'Predicted Rating'},
                color=errors,
                color_continuous_scale='Reds'
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
        
        with col2:
            fig = px.histogram(
                errors,
                nbins=30,
                title="Prediction Error Distribution",
                labels={'value': 'Absolute Error', 'count': 'Frequency'},
                color_discrete_sequence=['lightcoral']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        st.markdown("#### 📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sample RMSE", f"{np.sqrt(np.mean(errors**2)):.3f}")
        with col2:
            st.metric("Sample MAE", f"{np.mean(errors):.3f}")
        with col3:
            st.metric("Max Error", f"{np.max(errors):.3f}")
        with col4:
            st.metric("Min Error", f"{np.min(errors):.3f}")
    else:
        st.info("Train the model to see analysis")

with tab5:
    if 'model' in st.session_state:
        st.subheader("📊 Prediction Analysis")
        
        # Rating distribution analysis
        st.markdown("#### 📈 Rating Distribution Analysis")
        
        # Make predictions for all user-item pairs (sample)
        sample_users = np.random.choice(n_users, min(50, n_users), replace=False)
        sample_items = np.random.choice(n_items, min(50, n_items), replace=False)
        
        predictions_list = []
        for user in sample_users:
            X_pred = [np.array([user] * len(sample_items)), sample_items]
            preds = st.session_state['model'].predict(X_pred, verbose=0).flatten()
            preds = preds * (max_rating - min_rating) + min_rating
            predictions_list.extend(preds)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                predictions_list,
                nbins=30,
                title="Distribution of Predicted Ratings",
                labels={'value': 'Predicted Rating', 'count': 'Frequency'},
                color_discrete_sequence=['lightblue']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            actual_ratings = data['rating'].values
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=actual_ratings,
                name='Actual Ratings',
                opacity=0.6,
                marker_color='green'
            ))
            fig.add_trace(go.Histogram(
                x=predictions_list,
                name='Predicted Ratings',
                opacity=0.6,
                marker_color='blue'
            ))
            fig.update_layout(
                title="Actual vs Predicted Rating Distributions",
                xaxis_title="Rating",
                yaxis_title="Frequency",
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics comparison
        st.markdown("#### 📊 Distribution Statistics")
        
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median'],
            'Actual': [
                data['rating'].mean(),
                data['rating'].std(),
                data['rating'].min(),
                data['rating'].max(),
                data['rating'].median()
            ],
            'Predicted': [
                np.mean(predictions_list),
                np.std(predictions_list),
                np.min(predictions_list),
                np.max(predictions_list),
                np.median(predictions_list)
            ]
        })
        stats_df = stats_df.round(3)
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("Train the model to see prediction analysis")

# Footer
st.markdown("---")
st.markdown("### 💡 Algorithm Details")
with st.expander("Understanding Neural Collaborative Filtering"):
    st.markdown("""
    **Neural Collaborative Filtering:**
    
    This approach uses deep learning to learn complex user-item interactions:
    
    **1. Embeddings:**
    - Users and items are mapped to dense vectors (embeddings)
    - Similar users/items have similar embeddings
    - Embeddings capture latent features automatically
    
    **2. Architecture Options:**
    - **Dot Product**: Simple interaction between embeddings
    - **MLP**: Multi-layer perceptron for complex patterns
    - **Hybrid**: Combines both approaches
    
    **3. Training Process:**
    - Input: User ID, Item ID
    - Output: Predicted Rating
    - Loss: Mean Squared Error
    - Optimizer: Adam
    
    **Advantages:**
    - Learns non-linear patterns
    - Handles implicit feedback
    - Scalable to large datasets
    - No feature engineering required
    
    **Limitations:**
    - Requires sufficient data
    - Training can be slow
    - Less interpretable than traditional methods
    - Prone to overfitting
    
    **Hyperparameters:**
    - **Embedding Dimension**: Size of latent representations
    - **Architecture**: Model complexity
    - **Learning Rate**: Training speed
    - **Batch Size**: Training efficiency
    """)
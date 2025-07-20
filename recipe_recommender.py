import pandas as pd
import numpy as np
import pickle
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the datasets
df_recipes = pd.read_csv('recipes.csv')
df_test = pd.read_csv('test_recipes.csv')

# Data preprocessing function
def preprocess_data(df):
    # Fill NaN values
    df = df.fillna('')
    
    # Create a combined text field for embedding
    df['combined_text'] = df['recipe_name'] + ' ' + df['ingredients'] + ' ' + df['cuisine_path']
    
    # Clean text
    df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x).lower()))
    df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    return df

# Preprocess the main dataset
df_recipes = preprocess_data(df_recipes)

# Handle test dataset differently as it has a different structure
def preprocess_test_data(df):
    df = df.fillna('')
    
    # Convert string representation of lists to actual lists if needed
    def parse_ingredients(ing_str):
        if isinstance(ing_str, str):
            try:
                ingredients = eval(ing_str)
                # Extract ingredient names
                return ' '.join([item.get('name', '') for item in ingredients])
            except:
                return ing_str
        return ''
    
    # Apply parsing to ingredients column
    if 'Ingredients' in df.columns:
        df['ingredients'] = df['Ingredients'].apply(parse_ingredients)
    
    # Create combined text
    if 'Name' in df.columns and 'ingredients' in df.columns:
        df['combined_text'] = df['Name'] + ' ' + df['ingredients']
    else:
        # Fallback if columns are different
        df['combined_text'] = df.iloc[:, 1].astype(str) + ' ' + df.iloc[:, 7].astype(str)
    
    # Clean text
    df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x).lower()))
    df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    return df

# Preprocess test data
df_test = preprocess_test_data(df_test)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(df_recipes['combined_text'])

# Split data for neural network training
X_train, X_val, y_train, y_val = train_test_split(
    X_tfidf, 
    np.arange(len(df_recipes)),  # Using indices as targets for demonstration
    test_size=0.2, 
    random_state=42
)

# Create a nearest neighbors model for retrieval
knn_model = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine')
knn_model.fit(X_tfidf)

# Save the KNN model and vectorizer
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Tokenize text for neural network
max_words = 10000
max_seq_length = 200

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df_recipes['combined_text'])

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Convert text to sequences
X_sequences = tokenizer.texts_to_sequences(df_recipes['combined_text'])
X_padded = pad_sequences(X_sequences, maxlen=max_seq_length, padding='post')

# Split sequences for neural network
X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
    X_padded, 
    np.arange(len(df_recipes)),  # Using indices as targets
    test_size=0.2, 
    random_state=42
)

# Build the neural network with 6 different activation functions
# Input layer
input_layer = Input(shape=(max_seq_length,))

# Embedding layer
embed_dim = 128
embedding_layer = Embedding(input_dim=max_words, output_dim=embed_dim, input_length=max_seq_length)(input_layer)

# Bidirectional LSTM with tanh activation (activation function 1)
lstm_layer = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'))(embedding_layer)
dropout_1 = Dropout(0.3)(lstm_layer)

# Dense layer with ReLU activation (activation function 2)
dense_1 = Dense(256, activation='relu')(dropout_1)
dropout_2 = Dropout(0.3)(dense_1)

# Dense layer with LeakyReLU activation (activation function 3)
dense_2 = Dense(128)(dropout_2)
leaky_relu = LeakyReLU(alpha=0.1)(dense_2)
dropout_3 = Dropout(0.3)(leaky_relu)

# Dense layer with PReLU activation (activation function 4)
dense_3 = Dense(64)(dropout_3)
prelu = PReLU()(dense_3)
dropout_4 = Dropout(0.3)(prelu)

# Dense layer with ELU activation (activation function 5)
dense_4 = Dense(32)(dropout_4)
elu = ELU(alpha=1.0)(dense_4)

# Dense layer with ThresholdedReLU activation (activation function 6)
dense_5 = Dense(16)(elu)
thresholded_relu = ThresholdedReLU(theta=1.0)(dense_5)

# Global average pooling
global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(thresholded_relu)

# Output layer with softmax activation
output_layer = Dense(len(df_recipes), activation='softmax')(global_avg_pool)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with categorical crossentropy loss function
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # Loss function
    metrics=['accuracy']
)

# Print model summary
print(model.summary())

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=20,
    batch_size=64,
    callbacks=[early_stopping]
)

# Save the model
model.save('recipe_recommender_model.h5')

# Create a function to get recommendations
def get_recommendations(query, top_n=5):
    # Preprocess the query
    query = re.sub(r'[^\w\s]', ' ', query.lower())
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Transform query using TF-IDF vectorizer
    query_tfidf = tfidf_vectorizer.transform([query])
    
    # Get nearest neighbors using KNN
    distances, indices = knn_model.kneighbors(query_tfidf, n_neighbors=top_n)
    
    # Get the recommended recipes
    recommended_recipes = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        recipe = {
            'name': df_recipes.iloc[idx]['recipe_name'],
            'ingredients': df_recipes.iloc[idx]['ingredients'],
            'directions': df_recipes.iloc[idx]['directions'],
            'cuisine': df_recipes.iloc[idx]['cuisine_path'],
            'similarity_score': 1 - distances[0][i]  # Convert distance to similarity score
        }
        recommended_recipes.append(recipe)
    
    return recommended_recipes

# Create a function to get neural network based recommendations
def get_nn_recommendations(query, top_n=5):
    # Preprocess the query
    query = re.sub(r'[^\w\s]', ' ', query.lower())
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Convert query to sequence
    query_seq = tokenizer.texts_to_sequences([query])
    query_padded = pad_sequences(query_seq, maxlen=max_seq_length, padding='post')
    
    # Get predictions from neural network
    predictions = model.predict(query_padded)[0]
    
    # Get top N indices
    top_indices = np.argsort(predictions)[-top_n:]
    
    # Get the recommended recipes
    recommended_recipes = []
    for idx in top_indices:
        recipe = {
            'name': df_recipes.iloc[idx]['recipe_name'],
            'ingredients': df_recipes.iloc[idx]['ingredients'],
            'directions': df_recipes.iloc[idx]['directions'],
            'cuisine': df_recipes.iloc[idx]['cuisine_path'],
            'confidence': float(predictions[idx])
        }
        recommended_recipes.append(recipe)
    
    return recommended_recipes[::-1]  # Reverse to get highest confidence first

# Example usage
if __name__ == "__main__":
    # Test the recommendation system
    query = "chicken curry with rice"
    print("\nKNN-based recommendations for:", query)
    knn_recommendations = get_recommendations(query)
    for i, rec in enumerate(knn_recommendations):
        print(f"{i+1}. {rec['name']} (Similarity: {rec['similarity_score']:.2f})")
    
    print("\nNeural Network recommendations for:", query)
    nn_recommendations = get_nn_recommendations(query)
    for i, rec in enumerate(nn_recommendations):
        print(f"{i+1}. {rec['name']} (Confidence: {rec['confidence']:.4f})")
import pandas as pd
import numpy as np
from lottery_predictor import LotteryPredictor
import os
import ast
from collections import Counter
from sklearn.cluster import KMeans

def load_data(file_path):
    """Load historical lottery data from CSV"""
    df = pd.read_csv(file_path)
    
    # Convert date column if it exists
    if 'date' in df.columns:
        # Try parsing with the first format
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
        
        # For rows where parsing failed, try the second format
        mask = df['date'].isna()
        if mask.any():
            df.loc[mask, 'date'] = pd.to_datetime(df.loc[mask, 'date'], format='%H:%M %d-%m-%Y', errors='coerce')
    
    # Drop rows with NaNs in the 'numbers' column
    df = df.dropna(subset=['numbers'])
    
    # Convert 'numbers' column from string representation of lists to actual lists
    if 'numbers' in df.columns:
        df['numbers'] = df['numbers'].apply(ast.literal_eval)
    
    return df

def prepare_recent_draws(df, feature_columns):
    """Prepare recent draws for prediction ensuring feature alignment"""
    # Convert lists in 'numbers' column to individual columns
    numbers_df = pd.DataFrame(df['numbers'].tolist(), index=df.index)
    prepared_df = pd.concat([df.drop(columns=['numbers']), numbers_df], axis=1)

    # Ensure the columns match the training features
    for col in feature_columns:
        if col not in prepared_df.columns:
            prepared_df[col] = 0  # Add missing columns with default value 0

    # Reorder columns to match the training feature columns
    prepared_df = prepared_df[feature_columns]

    return prepared_df

def extract_sequence_patterns(draws, sequence_length=3):
    sequences = Counter()
    for draw in draws:
        numbers = draw['numbers']
        for i in range(len(numbers) - sequence_length + 1):
            sequence = tuple(numbers[i:i+sequence_length])
            sequences.update([sequence])
    return sequences.most_common()

def extract_clusters(draws, n_clusters=3):
    frequency = Counter()
    for draw in draws:
        frequency.update(draw['numbers'])
    numbers = list(frequency.keys())
    frequencies = list(frequency.values())
    X = np.array(frequencies).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    clusters = {i: [] for i in range(n_clusters)}
    for number, label in zip(numbers, kmeans.labels_):
        clusters[label].append(number)
    return clusters

def main():
    # Initialize predictor
    predictor = LotteryPredictor(numbers_range=(1, 80), numbers_to_draw=20)
    
    # Load historical data
    data_file = 'historical_draws.csv'
    print(f"Loading data from {data_file}...")
    historical_data = load_data(data_file)
    
    # Read the saved timestamp
    with open('model_timestamp.txt', 'r') as f:
        timestamp = f.read().strip()
    
    # Load models
    print("Loading existing ML models...")
    models_dir = 'ml_models'
    model_path = f'{models_dir}/lottery_predictor_{timestamp}'
    predictor.load_models(model_path)
    
    # Extract the feature columns used during training
    feature_columns = list(predictor.probabilistic_model.feature_names_in_)

    # Generate prediction for next draw
    print("\nGenerating prediction for next draw...")
    recent_draws = historical_data.tail(5).drop(columns=['date'])
    
    # Prepare recent draws for prediction with correct feature alignment
    recent_draws_prepared = prepare_recent_draws(recent_draws, feature_columns)
    
    # Generate prediction
    predicted_numbers, probabilities = predictor.predict(recent_draws_prepared)
    
    # Sort predicted numbers
    predicted_numbers_sorted = sorted(predicted_numbers)
    
    print("\nPredicted numbers for next draw:", predicted_numbers_sorted)
    
    # Print probability information for each number
    print("\nTop 10 most likely numbers and their probabilities:")
    number_probs = [(i+1, prob) for i, prob in enumerate(probabilities)]
    number_probs.sort(key=lambda x: x[1], reverse=True)
    for number, prob in number_probs[:10]:
        print(f"Number {number}: {prob:.4f}")

if __name__ == "__main__":
    main()
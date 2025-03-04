import sys
import os
import pandas as pd
import numpy as np
from lottery_predictor import LotteryPredictor
from datetime import datetime, timedelta
from collections import Counter
from sklearn.cluster import KMeans

print("Python path:", sys.path)
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found")
    
    df = pd.read_csv(file_path)
    try:
        df['date'] = pd.to_datetime(df['date'], format='%H:%M %d-%m-%Y', errors='coerce')
        df.loc[df['date'].isna(), 'date'] = pd.to_datetime(df.loc[df['date'].isna(), 'date'], errors='coerce')
    except Exception as e:
        print(f"Warning: Date conversion issue: {e}")
    
    number_cols = [f'number{i+1}' for i in range(20)]
    try:
        df[number_cols] = df[number_cols].astype(float)
    except Exception as e:
        print(f"Warning: Could not process number columns: {e}")
    
    for col in number_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    
    return df

def get_next_draw_time(current_time):
    minutes = (current_time.minute // 5 + 1) * 5
    next_draw_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)
    return next_draw_time

def save_predictions_to_csv(predicted_numbers, probabilities, next_draw_time, file_path):
    data = {
        'Timestamp': [next_draw_time.strftime('%H:%M %d-%m-%Y')],
        'Predicted Numbers': [','.join(map(str, predicted_numbers))],
        'Probabilities': [','.join(map(str, [probabilities[num - 1] for num in predicted_numbers]))]
    }
    df = pd.DataFrame(data)
    
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

def extract_sequence_patterns(draws, sequence_length=3):
    sequences = Counter()
    for draw in draws:
        numbers = [draw[f'number{i+1}'] for i in range(20)]
        for i in range(len(numbers) - sequence_length + 1):
            sequence = tuple(numbers[i:i+sequence_length])
            sequences.update([sequence])
    return sequences.most_common()

def extract_clusters(draws, n_clusters=3):
    frequency = Counter()
    for draw in draws:
        numbers = [draw[f'number{i+1}'] for i in range(20)]
        frequency.update(numbers)
    numbers = list(frequency.keys())
    frequencies = list(frequency.values())
    X = np.array(frequencies).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    clusters = {i: [] for i in range(n_clusters)}
    for number, label in zip(numbers, kmeans.labels_):
        clusters[label].append(number)
    return clusters

def main():
    try:
        predictor = LotteryPredictor(numbers_range=(1, 80), numbers_to_draw=20)
        
        models_dir = 'src/ml_models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        model_timestamp_file = 'src/model_timestamp.txt'
        model_loaded = False
        
        if os.path.exists(model_timestamp_file):
            with open(model_timestamp_file, 'r') as f:
                timestamp = f.read().strip()
                model_path = f'{models_dir}/lottery_predictor_{timestamp}'
                try:
                    predictor.load_models(model_path)
                    print(f"Model loaded from {model_path}")
                    model_loaded = True
                except Exception as e:
                    print(f"Error loading model: {e}")
        
        if not model_loaded:
            data_file = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\proiectnow\\Versiune1.4\\src\\historical_draws.csv'
            print(f"Loading data from {data_file}...")
            historical_data = load_data(data_file)
            
            print("Preparing training data...")
            X, y = predictor.prepare_data(historical_data)
            
            print(f"Shape of X: {X.shape}")
            print(f"Shape of y: {y.shape}")
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            print("Training models...")
            predictor.train_models(X_train, y_train)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f'{models_dir}/lottery_predictor_{timestamp}'
            predictor.save_models(model_path)
            
            with open(model_timestamp_file, 'w') as f:
                f.write(timestamp)
            
            print("\nModel training and saving complete.\n")
        else:
            data_file = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\proiectnow\\Versiune1.4\\src\\historical_draws.csv'
            print(f"Loading data from {data_file}...")
            historical_data = load_data(data_file)
        
        # Extract and display sequence patterns
        sequence_patterns = extract_sequence_patterns(historical_data.to_dict('records'))
        print("\nSequence Patterns:")
        for sequence, count in sequence_patterns:
            print(f"Sequence: {sequence}, Count: {count}")
        
        # Extract and display clusters
        clusters = extract_clusters(historical_data.to_dict('records'))
        print("\nClusters:")
        for cluster_id, cluster_numbers in clusters.items():
            print(f"Cluster {cluster_id}: {cluster_numbers}")
        
        print("Generating ML prediction for next draw...")
        recent_draws = historical_data.tail(5).copy()
        predicted_numbers, probabilities = predictor.predict(recent_draws)
        
        formatted_numbers = ','.join(map(str, predicted_numbers))
        next_draw_time = get_next_draw_time(datetime.now())
        print(f"Predicted numbers for the next draw at {next_draw_time.strftime('%H:%M %d-%m-%Y')}: {formatted_numbers}")
        print(f"Prediction probabilities: {[probabilities[num - 1] for num in predicted_numbers]}")
        
        predictions_file = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\proiectnow\\Versiune1.4\\data\\processed\\predictions.csv'
        save_predictions_to_csv(predicted_numbers, probabilities, next_draw_time, predictions_file)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if 'historical_data' in locals():
            print("\nFirst few rows of processed data:")
            print(historical_data.head())
            print("\nColumns in data:", historical_data.columns.tolist())

if __name__ == "__main__":
    main()
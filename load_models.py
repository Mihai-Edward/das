import pandas as pd
import numpy as np
from lottery_predictor import LotteryPredictor
import os
import ast
from collections import Counter
from sklearn.cluster import KMeans
from data_analysis import DataAnalysis
from datetime import datetime

class ModelLoader:
    def __init__(self):
        self.predictor = LotteryPredictor(numbers_range=(1, 80), numbers_to_draw=20)
        self.models_dir = 'src/ml_models'
        self.data_file = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\proiectnow\\Versiune1.4\\src\\historical_draws.csv'
        self.model_status = {
            'loaded': False,
            'timestamp': None,
            'features': None,
            'error': None
        }

    def load_data(self, file_path):
        """Load historical lottery data from CSV with enhanced error handling"""
        try:
            df = pd.read_csv(file_path)
            
            if 'date' in df.columns:
                # Try parsing with multiple formats
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
                mask = df['date'].isna()
                if mask.any():
                    df.loc[mask, 'date'] = pd.to_datetime(df.loc[mask, 'date'], 
                                                        format='%H:%M %d-%m-%Y', 
                                                        errors='coerce')
            
            # Ensure number columns are present
            number_cols = [f'number{i}' for i in range(1, 21)]
            for col in number_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

    def prepare_recent_draws(self, df, feature_columns):
        """Prepare recent draws with enhanced feature engineering"""
        try:
            # Basic feature preparation
            prepared_df = df.copy()
            
            # Add date-based features
            if 'date' in prepared_df.columns:
                prepared_df['day_of_week'] = prepared_df['date'].dt.dayofweek
                prepared_df['month'] = prepared_df['date'].dt.month
                prepared_df['day_of_year'] = prepared_df['date'].dt.dayofyear
                prepared_df['days_since_first_draw'] = (prepared_df['date'] - prepared_df['date'].min()).dt.days
            
            # Ensure all required features are present
            for col in feature_columns:
                if col not in prepared_df.columns:
                    prepared_df[col] = 0
            
            # Reorder columns to match training features
            prepared_df = prepared_df[feature_columns]
            
            return prepared_df
            
        except Exception as e:
            print(f"Error preparing draws: {str(e)}")
            return None

    def extract_sequence_patterns(self, draws, sequence_length=3):
        """Extract sequence patterns with validation"""
        try:
            sequences = Counter()
            for _, row in draws.iterrows():
                numbers = [row[f'number{i}'] for i in range(1, 21)]
                for i in range(len(numbers) - sequence_length + 1):
                    sequence = tuple(sorted(numbers[i:i+sequence_length]))
                    sequences.update([sequence])
            return sequences.most_common()
        except Exception as e:
            print(f"Error extracting sequences: {str(e)}")
            return []

    def extract_clusters(self, draws, n_clusters=3):
        """Extract clusters with enhanced analysis"""
        try:
            numbers = []
            frequencies = []
            
            # Count frequency of each number
            for i in range(1, 81):
                freq = sum(1 for _, row in draws.iterrows() 
                         if any(row[f'number{j}'] == i for j in range(1, 21)))
                numbers.append(i)
                frequencies.append(freq)
            
            X = np.array(frequencies).reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            
            clusters = {i: [] for i in range(n_clusters)}
            for number, label in zip(numbers, kmeans.labels_):
                clusters[label].append(number)
            
            return clusters
            
        except Exception as e:
            print(f"Error clustering numbers: {str(e)}")
            return None

    def load_models_with_validation(self):
        """Load models with enhanced validation and error handling"""
        try:
            # Check model timestamp
            timestamp_file = 'src/model_timestamp.txt'
            if not os.path.exists(timestamp_file):
                raise FileNotFoundError("Model timestamp file not found")
            
            with open(timestamp_file, 'r') as f:
                timestamp = f.read().strip()
            
            model_path = f'{self.models_dir}/lottery_predictor_{timestamp}'
            
            # Validate model files exist
            required_files = ['_prob_model.pkl', '_pattern_model.pkl', '_scaler.pkl']
            for file in required_files:
                if not os.path.exists(f"{model_path}{file}"):
                    raise FileNotFoundError(f"Missing model file: {file}")
            
            # Load models
            self.predictor.load_models(model_path)
            
            self.model_status.update({
                'loaded': True,
                'timestamp': timestamp,
                'features': list(self.predictor.probabilistic_model.feature_names_in_),
                'error': None
            })
            
            return True
            
        except Exception as e:
            self.model_status.update({
                'loaded': False,
                'error': str(e)
            })
            print(f"Error loading models: {str(e)}")
            return False

    def generate_prediction(self):
        """Generate prediction with enhanced analysis integration"""
        try:
            if not self.model_status['loaded']:
                if not self.load_models_with_validation():
                    return None, None, None
            
            # Load and prepare data
            historical_data = self.load_data(self.data_file)
            if historical_data is None:
                raise ValueError("Could not load historical data")
            
            # Get recent draws
            recent_draws = historical_data.tail(5)
            
            # Prepare features
            prepared_draws = self.prepare_recent_draws(
                recent_draws, 
                self.model_status['features']
            )
            
            if prepared_draws is None:
                raise ValueError("Could not prepare recent draws")
            
            # Generate prediction
            predicted_numbers, probabilities = self.predictor.predict(prepared_draws)
            
            # Generate analysis context
            analysis_context = {
                'sequences': self.extract_sequence_patterns(recent_draws),
                'clusters': self.extract_clusters(historical_data),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return predicted_numbers, probabilities, analysis_context
            
        except Exception as e:
            print(f"Error generating prediction: {str(e)}")
            return None, None, None

def main():
    loader = ModelLoader()
    
    # Load models and generate prediction
    predicted_numbers, probabilities, analysis = loader.generate_prediction()
    
    if predicted_numbers is not None:
        print("\n=== Prediction Results ===")
        print("Predicted numbers:", sorted(predicted_numbers))
        
        print("\nTop 10 most likely numbers and their probabilities:")
        number_probs = [(i+1, prob) for i, prob in enumerate(probabilities)]
        number_probs.sort(key=lambda x: x[1], reverse=True)
        for number, prob in number_probs[:10]:
            print(f"Number {number}: {prob:.4f}")
            
        if analysis:
            print("\n=== Analysis Context ===")
            print("Common sequences:", analysis['sequences'][:5])
            print("Number clusters:", analysis['clusters'])
    else:
        print("\nFailed to generate prediction")
        print(f"Model Status: {loader.model_status}")

if __name__ == "__main__":
    main()
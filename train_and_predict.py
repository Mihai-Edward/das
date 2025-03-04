import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lottery_predictor import LotteryPredictor
from datetime import datetime, timedelta
from collections import Counter
from sklearn.cluster import KMeans
from data_analysis import DataAnalysis

class TrainPredictor:
    def __init__(self):
        self.predictor = LotteryPredictor(numbers_range=(1, 80), numbers_to_draw=20)
        self.models_dir = 'src/ml_models'
        self.data_file = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\proiectnow\\Versiune1.4\\src\\historical_draws.csv'
        self.predictions_file = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\proiectnow\\Versiune1.4\\data\\processed\\predictions.csv'
        self.training_status = {
            'success': False,
            'model_loaded': False,
            'timestamp': None,
            'error': None
        }
        self.data = None

    def clean_data(self, data):
        """Clean the lottery draw data"""
        try:
            # Remove any duplicates
            data = data.drop_duplicates()
            
            # Ensure numbers are within valid range (1-80)
            data = data[(data >= 1) & (data <= 80)]
            
            # Sort numbers in each draw
            data = np.sort(data)
            
            # Remove any rows with missing values
            data = data.dropna()
            
            print("Data cleaning completed successfully")
            return data
            
        except Exception as e:
            print(f"Error during data cleaning: {e}")
            return data  # Return original data if cleaning fails

    def load_data(self, file_path):
        """Enhanced data loading with validation"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found")
        
        df = pd.read_csv(file_path)
        try:
            df['date'] = pd.to_datetime(df['date'], format='%H:%M %d-%m-%Y', errors='coerce')
            df.loc[df['date'].isna(), 'date'] = pd.to_datetime(df.loc[df['date'].isna(), 'date'], errors='coerce')
        except Exception as e:
            print(f"Warning: Date conversion issue: {e}")
        
        number_cols = [f'number{i+1}' for i in range(20)]
        df[number_cols] = df[number_cols].astype(float)
        
        # Enhanced validation
        for col in number_cols:
            if col in df.columns:
                invalid_numbers = df[~df[col].between(1, 80)][col]
                if not invalid_numbers.empty:
                    print(f"Warning: Found invalid numbers in {col}")
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        
        return df

    def prepare_data(self):
        """Load and clean the data"""
        self.data = self.load_data(self.data_file)
        self.data = self.clean_data(self.data)
        # Continue with your existing processing...

    def extract_sequence_patterns(self, draws, sequence_length=3):
        """Enhanced pattern extraction with validation"""
        sequences = Counter()
        try:
            for draw in draws:
                numbers = [draw[f'number{i+1}'] for i in range(20)]
                if all(isinstance(n, (int, float)) and 1 <= n <= 80 for n in numbers):
                    for i in range(len(numbers) - sequence_length + 1):
                        sequence = tuple(sorted(numbers[i:i+sequence_length]))
                        sequences.update([sequence])
            return sequences.most_common()
        except Exception as e:
            print(f"Error in sequence pattern extraction: {e}")
            return []

    def extract_clusters(self, draws, n_clusters=3):
        """Enhanced clustering with validation"""
        try:
            frequency = Counter()
            for draw in draws:
                numbers = [draw[f'number{i+1}'] for i in range(20)]
                valid_numbers = [n for n in numbers if isinstance(n, (int, float)) and 1 <= n <= 80]
                frequency.update(valid_numbers)
            
            numbers = list(frequency.keys())
            frequencies = list(frequency.values())
            X = np.array(frequencies).reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            
            clusters = {i: [] for i in range(n_clusters)}
            for number, label in zip(numbers, kmeans.labels_):
                clusters[label].append(number)
            return clusters
        except Exception as e:
            print(f"Error in clustering: {e}")
            return None

    def train_or_load_model(self):
        """Enhanced model training and loading"""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            model_timestamp_file = 'src/model_timestamp.txt'
            
            if os.path.exists(model_timestamp_file):
                with open(model_timestamp_file, 'r') as f:
                    timestamp = f.read().strip()
                    model_path = f'{self.models_dir}/lottery_predictor_{timestamp}'
                    try:
                        self.predictor.load_models(model_path)
                        print(f"Model loaded from {model_path}")
                        self.training_status.update({
                            'model_loaded': True,
                            'timestamp': timestamp
                        })
                        return True
                    except Exception as e:
                        print(f"Error loading model: {e}")
            
            # Train new model
            historical_data = self.load_data(self.data_file)
            
            # Add analysis features
            analyzer = DataAnalysis([])
            analysis_features = self.get_analysis_features(historical_data)
            
            X, y = self.predictor.prepare_data(historical_data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.predictor.train_models(X_train, y_train)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f'{self.models_dir}/lottery_predictor_{timestamp}'
            self.predictor.save_models(model_path)
            
            with open(model_timestamp_file, 'w') as f:
                f.write(timestamp)
            
            self.training_status.update({
                'success': True,
                'timestamp': timestamp,
                'model_loaded': True
            })
            
            return True
            
        except Exception as e:
            self.training_status.update({
                'success': False,
                'error': str(e)
            })
            print(f"Error in model training: {e}")
            return False

    def get_analysis_features(self, data):
        """Get enhanced analysis features"""
        try:
            draws = [(row['date'].strftime('%H:%M %d-%m-%Y'), 
                     [row[f'number{i}'] for i in range(1, 21)]) 
                    for _, row in data.iterrows()]
            
            analyzer = DataAnalysis(draws)
            return {
                'frequency': analyzer.count_frequency(),
                'hot_cold': analyzer.hot_and_cold_numbers(),
                'common_pairs': analyzer.find_common_pairs(),
                'range_analysis': analyzer.number_range_analysis(),
                'sequences': self.extract_sequence_patterns(data.to_dict('records')),
                'clusters': self.extract_clusters(data.to_dict('records'))
            }
        except Exception as e:
            print(f"Error getting analysis features: {e}")
            return {}

    def generate_prediction(self):
        """Enhanced prediction generation"""
        try:
            if not self.training_status['model_loaded']:
                if not self.train_or_load_model():
                    return None, None, None
            
            historical_data = self.load_data(self.data_file)
            recent_draws = historical_data.tail(5).copy()
            
            # Generate prediction with analysis context
            predicted_numbers, probabilities = self.predictor.predict(recent_draws)
            analysis_context = self.get_analysis_features(historical_data)
            
            next_draw_time = self.get_next_draw_time(datetime.now())
            
            # Save prediction
            self.save_predictions_to_csv(predicted_numbers, probabilities, next_draw_time)
            
            return predicted_numbers, probabilities, analysis_context
            
        except Exception as e:
            print(f"Error generating prediction: {e}")
            return None, None, None

    def get_next_draw_time(self, current_time):
        """Get next draw time"""
        minutes = (current_time.minute // 5 + 1) * 5
        return current_time.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)

    def save_predictions_to_csv(self, predicted_numbers, probabilities, next_draw_time):
        """Enhanced prediction saving"""
        try:
            data = {
                'Timestamp': [next_draw_time.strftime('%H:%M %d-%m-%Y')],
                'Predicted_Numbers': [','.join(map(str, predicted_numbers))],
                'Probabilities': [','.join(map(str, [probabilities[num - 1] for num in predicted_numbers]))]
            }
            df = pd.DataFrame(data)
            
            os.makedirs(os.path.dirname(self.predictions_file), exist_ok=True)
            
            if os.path.exists(self.predictions_file):
                df.to_csv(self.predictions_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.predictions_file, index=False)
                
        except Exception as e:
            print(f"Error saving predictions: {e}")

def main():
    predictor = TrainPredictor()
    
    try:
        predictor.prepare_data()
        predicted_numbers, probabilities, analysis = predictor.generate_prediction()
        
        if predicted_numbers is not None:
            print("\n=== Prediction Results ===")
            print(f"Predicted numbers: {sorted(predicted_numbers)}")
            
            if analysis:
                print("\n=== Analysis Context ===")
                print("Hot numbers:", analysis.get('hot_cold', ('N/A', 'N/A'))[0])
                print("Common pairs:", list(analysis.get('common_pairs', {}).items())[:5])
                print("Range analysis:", analysis.get('range_analysis', {}))
        
        else:
            print("\nFailed to generate prediction")
            print(f"Training status: {predictor.training_status}")
            
    except Exception as e:
        print(f"\nError in main execution: {e}")

if __name__ == "__main__":
    main()
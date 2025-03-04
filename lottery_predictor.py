import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
import joblib
from collections import OrderedDict, Counter
from data_analysis import DataAnalysis
from datetime import datetime, timedelta
import os
import glob
from sklearn.model_selection import train_test_split

class LotteryPredictor:
    def __init__(self, numbers_range=(1, 80), numbers_to_draw=20):
        # Core settings
        self.numbers_range = numbers_range
        self.numbers_to_draw = numbers_to_draw
        
        # File paths (preserved from original)
        self.models_dir = 'src/ml_models'
        self.data_file = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\proiectnow\\Versiune1.4\\src\\historical_draws.csv'
        self.predictions_file = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\proiectnow\\Versiune1.4\\data\\processed\\predictions.csv'
        
        # Models initialization
        self.scaler = StandardScaler()
        self.probabilistic_model = GaussianNB()
        self.pattern_model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            max_iter=500
        )
        
        # Analysis components
        self.analyzer = DataAnalysis([])
        
        # Enhanced state tracking
        self.training_status = {
            'success': False,
            'model_loaded': False,
            'timestamp': None,
            'error': None,
            'prob_score': None,
            'pattern_score': None,
            'features': None
        }
        
        # Initialize pipeline
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the prediction pipeline with ordered stages"""
        self.pipeline_stages = OrderedDict({
            'data_preparation': self._prepare_pipeline_data,
            'feature_engineering': self._create_enhanced_features,
            'model_prediction': self._generate_model_predictions,
            'post_processing': self._post_process_predictions
        })
        self.pipeline_data = {}
    
    def _prepare_pipeline_data(self, data):
        """First pipeline stage: Prepare and validate input data"""
        print("\nPreparing data for prediction pipeline...")
        try:
            # Ensure we have a DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            # Ensure we have the required number of rows (5 recent draws)
            if len(data) < 5:
                raise ValueError("Need at least 5 recent draws for prediction")
                
            # Clean the data
            prepared_data = self.clean_data(data)
            if prepared_data is None or len(prepared_data) < 5:
                raise ValueError("Data cleaning resulted in insufficient data")
                
            # Ensure all required columns exist
            required_cols = ['date'] + [f'number{i+1}' for i in range(20)]
            missing_cols = [col for col in required_cols if col not in prepared_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Sort by date and get most recent 5 draws
            prepared_data = prepared_data.sort_values('date').tail(5)
            
            # Validate number ranges
            number_cols = [f'number{i+1}' for i in range(20)]
            for col in number_cols:
                invalid_numbers = prepared_data[~prepared_data[col].between(1, 80)]
                if not invalid_numbers.empty:
                    raise ValueError(f"Invalid numbers found in column {col}")
                    
            # Add date-based features
            if 'date' in prepared_data.columns:
                prepared_data['day_of_week'] = prepared_data['date'].dt.dayofweek
                prepared_data['month'] = prepared_data['date'].dt.month
                prepared_data['day_of_year'] = prepared_data['date'].dt.dayofyear
                prepared_data['days_since_first_draw'] = (
                    prepared_data['date'] - prepared_data['date'].min()
                ).dt.days
                
                # Store additional features in pipeline data
                self.pipeline_data['date_features'] = {
                    'day_of_week': prepared_data['day_of_week'].tolist(),
                    'month': prepared_data['month'].tolist(),
                    'day_of_year': prepared_data['day_of_year'].tolist(),
                    'days_since_first_draw': prepared_data['days_since_first_draw'].tolist()
                }
            
            # Store in pipeline data for potential later use
            self.pipeline_data['prepared_data'] = prepared_data
            print("Data preparation completed successfully")
            return prepared_data
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return None

    def load_data(self, file_path=None):
        """Enhanced data loading with validation"""
        if file_path is None:
            file_path = self.data_file
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found")
        
        try:
            df = pd.read_csv(file_path)
            
            # Enhanced date parsing with multiple formats
            try:
                df['date'] = pd.to_datetime(df['date'], format='%H:%M %d-%m-%Y', errors='coerce')
                df.loc[df['date'].isna(), 'date'] = pd.to_datetime(
                    df.loc[df['date'].isna(), 'date'], 
                    errors='coerce'
                )
            except Exception as e:
                print(f"Warning: Date conversion issue: {e}")
            
            # Convert number columns to float
            number_cols = [f'number{i+1}' for i in range(20)]
            df[number_cols] = df[number_cols].astype(float)
            
            # Validate number ranges
            for col in number_cols:
                if col in df.columns:
                    invalid_numbers = df[~df[col].between(1, 80)][col]
                    if not invalid_numbers.empty:
                        print(f"Warning: Found invalid numbers in {col}")
                    df[col] = df[col].fillna(
                        df[col].mode()[0] if not df[col].mode().empty else 0
                    )
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def clean_data(self, data):
        """Enhanced data cleaning with validation"""
        try:
            # Remove duplicates
            data = data.drop_duplicates()
            
            # Ensure numbers are within valid range (1-80)
            number_cols = [f'number{i+1}' for i in range(20)]
            for col in number_cols:
                data = data[(data[col] >= 1) & (data[col] <= 80)]
            
            # Sort numbers in each draw
            for _, row in data.iterrows():
                numbers = sorted(row[number_cols])
                for i, num in enumerate(numbers):
                    data.at[row.name, f'number{i+1}'] = num
            
            # Remove rows with missing values
            data = data.dropna()
            
            print("Data cleaning completed successfully")
            return data
            
        except Exception as e:
            print(f"Error during data cleaning: {e}")
            return data

    def prepare_data(self, historical_data):
        """Prepare data for training"""
        historical_data = historical_data.sort_values('date')
        features = []
        labels = []
        
        for i in range(len(historical_data) - 5):
            window = historical_data.iloc[i:i+5]
            next_draw = historical_data.iloc[i+5]
            feature_vector = self._create_feature_vector(window)
            features.append(feature_vector)
            labels.append(next_draw['number1'])
            
            print(f"Processing next_draw: {next_draw[[f'number{i+1}' for i in range(20)]]}")
        
        features = np.array(features)
        labels = np.array(labels)
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        return features, labels

    def _create_feature_vector(self, window):
        """Create base feature vector"""
        features = []
        number_counts = np.zeros(self.numbers_range[1] - self.numbers_range[0] + 1)
        
        # Count number frequencies
        number_cols = [f'number{i+1}' for i in range(20)]
        for _, row in window.iterrows():
            for num in row[number_cols]:
                number_counts[int(num) - self.numbers_range[0]] += 1
                
        features.extend(number_counts / len(window))
        print(f"Window features: {features}")
        
        # Add statistical features
        last_draw = window.iloc[-1][number_cols]
        features.extend([
            np.mean(last_draw),
            np.std(last_draw),
            len(set(last_draw) & set(range(1, 41))),
            len(set(last_draw) & set(range(41, 81)))
        ])
        
        print(f"Feature vector: {features}")
        return np.array(features)
    
    def _create_analysis_features(self, data):
        """Create enhanced features from data analysis"""
        print("\nGenerating analysis features...")
        
        try:
            # Update analyzer with current data
            formatted_draws = [(row['date'], 
                              [row[f'number{i}'] for i in range(1, 21)]) 
                             for _, row in data.iterrows()]
            self.analyzer = DataAnalysis(formatted_draws)
            
            # Get analysis results
            frequency = self.analyzer.count_frequency()
            hot_numbers, cold_numbers = self.analyzer.hot_and_cold_numbers()
            common_pairs = self.analyzer.find_common_pairs()
            range_analysis = self.analyzer.number_range_analysis()
            sequences = self.extract_sequence_patterns(data)
            clusters = self.extract_clusters(data)
            
            # Convert analysis results to features
            analysis_features = []
            
            # Frequency features
            freq_vector = np.zeros(80)
            for num, freq in frequency.items():
                freq_vector[num-1] = freq
            analysis_features.extend(freq_vector / np.sum(freq_vector))
            
            # Hot/Cold numbers features
            hot_vector = np.zeros(80)
            for num, _ in hot_numbers:
                hot_vector[num-1] = 1
            analysis_features.extend(hot_vector)
            
            # Common pairs features
            pairs_vector = np.zeros(80)
            for (num1, num2), freq in common_pairs:
                pairs_vector[num1-1] += freq
                pairs_vector[num2-1] += freq
            analysis_features.extend(pairs_vector / np.sum(pairs_vector))
            
            # Range analysis features
            range_vector = np.zeros(4)
            for i, (range_name, count) in enumerate(range_analysis.items()):
                range_vector[i] = count
            analysis_features.extend(range_vector / np.sum(range_vector))
            
            # Store analysis context
            self.pipeline_data['analysis_context'] = {
                'frequency': frequency,
                'hot_cold': (hot_numbers, cold_numbers),
                'common_pairs': common_pairs,
                'range_analysis': range_analysis,
                'sequences': sequences,
                'clusters': clusters
            }
            
            return np.array(analysis_features)
            
        except Exception as e:
            print(f"Error in analysis features generation: {e}")
            return np.zeros(244)  # Return zero vector of expected size

    def extract_sequence_patterns(self, data, sequence_length=3):
        """Extract sequence patterns with validation"""
        try:
            sequences = Counter()
            for _, row in data.iterrows():
                numbers = [row[f'number{i+1}'] for i in range(20)]
                if all(isinstance(n, (int, float)) and 1 <= n <= 80 for n in numbers):
                    for i in range(len(numbers) - sequence_length + 1):
                        sequence = tuple(sorted(numbers[i:i+sequence_length]))
                        sequences.update([sequence])
            return sequences.most_common()
            
        except Exception as e:
            print(f"Error in sequence pattern extraction: {e}")
            return []

    def extract_clusters(self, data, n_clusters=3):
        """Extract clusters with enhanced validation"""
        try:
            frequency = Counter()
            for _, row in data.iterrows():
                numbers = [row[f'number{i+1}'] for i in range(20)]
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

    def get_analysis_features(self, data):
        """Get enhanced analysis features from TrainPredictor"""
        try:
            draws = [(row['date'].strftime('%H:%M %d-%m-%Y'), 
                     [row[f'number{i}'] for i in range(1, 21)]) 
                    for _, row in data.iterrows()]
            
            analyzer = DataAnalysis(draws)
            analysis_features = {
                'frequency': analyzer.count_frequency(),
                'hot_cold': analyzer.hot_and_cold_numbers(),
                'common_pairs': analyzer.find_common_pairs(),
                'range_analysis': analyzer.number_range_analysis(),
                'sequences': self.extract_sequence_patterns(data),
                'clusters': self.extract_clusters(data)
            }
            
            # Store in pipeline data
            self.pipeline_data['analysis_features'] = analysis_features
            return analysis_features
            
        except Exception as e:
            print(f"Error getting analysis features: {e}")
            return {}

    def _create_enhanced_features(self, data):
        """Create enhanced feature vector combining all features"""
        print("\nGenerating enhanced features...")
        try:
            # Get base features
            base_features = self._create_feature_vector(data)
            
            # Get analysis features
            analysis_features = self._create_analysis_features(data)
            
            # Combine features
            enhanced_features = np.concatenate([base_features, analysis_features])
            print(f"Enhanced feature vector shape: {enhanced_features.shape}")
            
            self.pipeline_data['features'] = enhanced_features
            return enhanced_features
            
        except Exception as e:
            print(f"Error in enhanced feature creation: {e}")
            return self._create_feature_vector(data)  # Fallback to base features
    
    def _generate_model_predictions(self, features):
        """Generate predictions from both models"""
        print("\nGenerating model predictions...")
        try:
            scaled_features = self.scaler.transform([features])
            
            prob_pred = self.probabilistic_model.predict_proba(scaled_features)[0]
            pattern_pred = self.pattern_model.predict_proba(scaled_features)[0]
            
            self.pipeline_data['prob_pred'] = prob_pred
            self.pipeline_data['pattern_pred'] = pattern_pred
            return prob_pred, pattern_pred
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return None, None

    def _post_process_predictions(self, predictions):
        """Process and combine predictions"""
        print("\nPost-processing predictions...")
        try:
            if predictions[0] is None or predictions[1] is None:
                raise ValueError("Invalid predictions received")
                
            prob_pred, pattern_pred = predictions
            combined_pred = 0.4 * prob_pred + 0.6 * pattern_pred
            
            predicted_numbers = np.argsort(combined_pred)[-self.numbers_to_draw:]
            final_numbers = sorted([int(i) + self.numbers_range[0] for i in predicted_numbers])
            
            self.pipeline_data['final_prediction'] = final_numbers
            self.pipeline_data['probabilities'] = combined_pred
            return final_numbers, combined_pred
            
        except Exception as e:
            print(f"Error in post-processing: {e}")
            return None, None

    def save_models(self, path_prefix=None):
        """Save models with timestamp"""
        try:
            if path_prefix is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path_prefix = f'{self.models_dir}/lottery_predictor_{timestamp}'
            
            os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
            
            joblib.dump(self.probabilistic_model, f'{path_prefix}_prob_model.pkl')
            joblib.dump(self.pattern_model, f'{path_prefix}_pattern_model.pkl')
            joblib.dump(self.scaler, f'{path_prefix}_scaler.pkl')
            
            # Update timestamp file
            with open('src/model_timestamp.txt', 'w') as f:
                f.write(path_prefix.split('_')[-1])
            
            return True
            
        except Exception as e:
            print(f"Error saving models: {e}")
            return False

    def load_models(self, path_prefix=None):
        """Enhanced model loading with validation from ModelLoader"""
        try:
            if path_prefix is None:
                # Get latest model
                model_files = glob.glob(f"{self.models_dir}/*_prob_model.pkl")
                if not model_files:
                    raise FileNotFoundError("No models found")
                path_prefix = max(model_files, key=os.path.getctime).replace('_prob_model.pkl', '')
            
            # Validate required files exist
            required_files = ['_prob_model.pkl', '_pattern_model.pkl', '_scaler.pkl']
            for file in required_files:
                if not os.path.exists(f"{path_prefix}{file}"):
                    raise FileNotFoundError(f"Missing model file: {file}")
            
            # Load models
            self.probabilistic_model = joblib.load(f'{path_prefix}_prob_model.pkl')
            self.pattern_model = joblib.load(f'{path_prefix}_pattern_model.pkl')
            self.scaler = joblib.load(f'{path_prefix}_scaler.pkl')
            
            # Update status with feature information
            self.training_status.update({
                'model_loaded': True,
                'timestamp': datetime.fromtimestamp(
                    os.path.getctime(f'{path_prefix}_prob_model.pkl')
                ),
                'features': getattr(self.probabilistic_model, 'feature_names_in_', None)
            })
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.training_status['error'] = str(e)
            return False

    def train_and_predict(self, historical_data=None, recent_draws=None):
        """Enhanced prediction generation with analysis integration"""
        try:
            if historical_data is None:
                historical_data = self.load_data()
            
            if recent_draws is None:
                recent_draws = historical_data.tail(5)
            
            # Check if model needs training
            if not self.training_status['model_loaded']:
                print("\nTraining new model...")
                X, y = self.prepare_data(historical_data)
                
                # Add analysis features
                analysis_features = self.get_analysis_features(historical_data)
                
                # Train models
                self.train_models(X, y)
                self.save_models()
            
            # Generate prediction with analysis context
            predicted_numbers, probabilities, analysis_context = self.predict(recent_draws)
            
            # Get next draw time and save prediction
            next_draw_time = datetime.now().replace(
                minute=(datetime.now().minute // 5 + 1) * 5,
                second=0, 
                microsecond=0
            )
            
            if predicted_numbers is not None:
                self.save_prediction_to_csv(predicted_numbers, probabilities)
            
            return predicted_numbers, probabilities, analysis_context
            
        except Exception as e:
            print(f"Error in train_and_predict: {e}")
            return None, None, None

    def train_models(self, X_train, y_train):
        """Train both models with test split"""
        try:
            # Add train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            self.probabilistic_model.fit(X_train_scaled, y_train)
            self.pattern_model.fit(X_train_scaled, y_train)
            
            # Add model evaluation
            self.training_status.update({
                'success': True,
                'model_loaded': True,
                'timestamp': datetime.now(),
                'prob_score': self.probabilistic_model.score(X_test_scaled, y_test),
                'pattern_score': self.pattern_model.score(X_test_scaled, y_test)
            })
            return True
        except Exception as e:
            print(f"Error training models: {e}")
            self.training_status.update({
                'success': False,
                'error': str(e)
            })
            return False

    def predict(self, recent_draws):
        """Enhanced prediction with pipeline execution"""
        try:
            # Run prediction pipeline
            result = recent_draws
            for stage_name, stage_func in self.pipeline_stages.items():
                print(f"\nExecuting pipeline stage: {stage_name}")
                result = stage_func(result)
                
                if result is None:
                    raise ValueError(f"Pipeline stage {stage_name} failed")
            
            # Get final predictions and analysis
            final_numbers = self.pipeline_data['final_prediction']
            probabilities = self.pipeline_data['probabilities']
            analysis_context = self.pipeline_data.get('analysis_context', {})
            
            return final_numbers, probabilities, analysis_context
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, None, None

    def save_prediction_to_csv(self, predicted_numbers, probabilities):
        """Save prediction results"""
        try:
            next_draw_time = datetime.now().replace(
                minute=(datetime.now().minute // 5 + 1) * 5,
                second=0, 
                microsecond=0
            )
            
            data = {
                'Timestamp': [next_draw_time.strftime('%H:%M %d-%m-%Y')],
                'Predicted_Numbers': [','.join(map(str, predicted_numbers))],
                'Probabilities': [','.join(map(str, [probabilities[num - 1] for num in predicted_numbers]))]
            }
            
            df = pd.DataFrame(data)
            os.makedirs(os.path.dirname(self.predictions_file), exist_ok=True)
            
            mode = 'a' if os.path.exists(self.predictions_file) else 'w'
            header = not os.path.exists(self.predictions_file)
            df.to_csv(self.predictions_file, mode=mode, header=header, index=False)
            
        except Exception as e:
            print(f"Error saving prediction: {e}")

    def prepare_feature_columns(self, data):
        """Ensure all required feature columns exist"""
        try:
            if self.training_status['model_loaded'] and hasattr(self.probabilistic_model, 'feature_names_in_'):
                required_features = self.probabilistic_model.feature_names_in_
                for col in required_features:
                    if col not in data.columns:
                        data[col] = 0
                return data[required_features]
            return data
        except Exception as e:
            print(f"Error preparing feature columns: {e}")
            return data

if __name__ == "__main__":
    predictor = LotteryPredictor()
    
    try:
        # Load historical data
        historical_data = predictor.load_data()
        if historical_data is None:
            raise ValueError("Failed to load historical data")
        
        # Generate prediction
        prediction, probabilities, analysis = predictor.train_and_predict(
            historical_data=historical_data
        )
        
        if prediction is not None:
            print("\n=== Prediction Results ===")
            print(f"Predicted numbers: {sorted(prediction)}")
            print("\n=== Analysis Context ===")
            for key, value in analysis.items():
                print(f"{key}: {value}")
            
            # Save prediction
            predictor.save_prediction_to_csv(prediction, probabilities)
        
    except Exception as e:
        print(f"\nError in main execution: {e}")
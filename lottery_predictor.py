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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import PATHS, ensure_directories

class LotteryPredictor:
    def __init__(self, numbers_range=(1, 80), numbers_to_draw=20):
        """Initialize LotteryPredictor with enhanced model configuration"""
        # Validate input parameters
        if not isinstance(numbers_range, tuple) or len(numbers_range) != 2:
            raise ValueError("numbers_range must be a tuple of (min, max)")
        if not isinstance(numbers_to_draw, int) or numbers_to_draw <= 0:
            raise ValueError("numbers_to_draw must be a positive integer")
            
        # Core settings with validation
        self.numbers_range = numbers_range
        self.numbers_to_draw = numbers_to_draw
        self.num_classes = numbers_range[1] - numbers_range[0] + 1
        
        # Initialize paths using config
        ensure_directories()
        self.models_dir = PATHS['MODELS_DIR']
        self.data_file = PATHS['HISTORICAL_DATA']
        self.predictions_file = PATHS['PREDICTIONS']
        
        # Models initialization with enhanced configuration
        self.scaler = StandardScaler()
        
        # Probabilistic model for multi-class classification
        self.probabilistic_model = GaussianNB(
            priors=None,  # Let the model learn class probabilities from data
            var_smoothing=1e-9  # Default smoothing parameter
        )
        
        # Neural network with optimized architecture
        self.pattern_model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 80),  # Optimized for lottery number prediction
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization parameter
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            shuffle=True,
            random_state=42,
            early_stopping=True,  # Enable early stopping
            validation_fraction=0.1,  # Use 10% of training data for validation
            n_iter_no_change=10  # Number of iterations with no improvement to wait before stopping
        )
        
        # Initialize analysis components
        self.analyzer = None  # Will be initialized when needed
        
        # Enhanced state tracking
        self.training_status = {
            'success': False,
            'model_loaded': False,
            'timestamp': None,
            'error': None,
            'prob_score': None,
            'pattern_score': None,
            'features': None,
            'model_config': {
                'num_classes': self.num_classes,
                'numbers_to_draw': self.numbers_to_draw,
                'feature_dimension': None  # Will be set during training
            }
        }
        
        # Initialize pipeline data storage
        self.pipeline_data = {}
        
        print(f"\nInitialized LotteryPredictor:")
        print(f"- Number range: {numbers_range}")
        print(f"- Numbers to draw: {numbers_to_draw}")
        print(f"- Number of classes: {self.num_classes}")
    
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
            formatted_draws = []
            for _, row in data.iterrows():
                numbers = []
                for i in range(1, 21):
                    num = row.get(f'number{i}')
                    if isinstance(num, (int, float)) and 1 <= num <= 80:
                        numbers.append(int(num))
                if len(numbers) == 20:
                    formatted_draws.append((row['date'], numbers))
            
            self.analyzer = DataAnalysis(formatted_draws)
            
            # Get analysis results
            frequency = self.analyzer.count_frequency()
            hot_numbers, cold_numbers = self.analyzer.hot_and_cold_numbers()
            common_pairs = self.analyzer.find_common_pairs()
            range_analysis = self.analyzer.number_range_analysis()
            
            # Convert analysis results to features - fixed size array
            analysis_features = np.zeros(160)  # 80 for frequency + 80 for hot/cold
            
            # Frequency features (first 80)
            total_freq = sum(frequency.values()) or 1  # Avoid division by zero
            for num, freq in frequency.items():
                if 1 <= num <= 80:
                    analysis_features[num-1] = freq / total_freq
            
            # Hot/Cold features (second 80)
            hot_nums = dict(hot_numbers)
            max_hot_score = max(hot_nums.values()) if hot_nums else 1
            for num, score in hot_nums.items():
                if 1 <= num <= 80:
                    analysis_features[80 + num-1] = score / max_hot_score
            
            # Store analysis context with additional metadata
            self.pipeline_data['analysis_context'] = {
                'frequency': frequency,
                'hot_cold': (hot_numbers, cold_numbers),
                'common_pairs': common_pairs,
                'range_analysis': range_analysis,
                'feature_stats': {
                    'total_frequency': total_freq,
                    'max_hot_score': max_hot_score,
                    'feature_range': (np.min(analysis_features), np.max(analysis_features))
                }
            }
            
            print(f"Generated {len(analysis_features)} analysis features")
            print(f"Feature range: {np.min(analysis_features):.4f} to {np.max(analysis_features):.4f}")
            
            return analysis_features
            
        except Exception as e:
            print(f"Error in analysis features generation: {e}")
            return np.zeros(160)  # Return zero vector of fixed size

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
            # Get base features (80 frequency features + 4 statistical features)
            base_features = self._create_feature_vector(data)
            
            # Get analysis features (160 features)
            analysis_features = self._create_analysis_features(data)
            
            # Validate feature dimensions
            if base_features is None or analysis_features is None:
                raise ValueError("Failed to generate either base or analysis features")
            
            print(f"Base features shape: {base_features.shape}")
            print(f"Analysis features shape: {analysis_features.shape}")
            
            # Ensure consistent feature dimensions
            if len(base_features) == 84 and len(analysis_features) == 160:
                # Store both feature sets in pipeline data
                self.pipeline_data['base_features'] = base_features
                self.pipeline_data['analysis_features'] = analysis_features
                
                # For now, use only base features for prediction
                # This ensures compatibility with existing trained models
                enhanced_features = base_features
                
                # Add feature metadata
                self.pipeline_data['feature_metadata'] = {
                    'base_features_size': len(base_features),
                    'analysis_features_size': len(analysis_features),
                    'used_features_size': len(enhanced_features),
                    'feature_stats': {
                        'mean': float(np.mean(enhanced_features)),
                        'std': float(np.std(enhanced_features)),
                        'min': float(np.min(enhanced_features)),
                        'max': float(np.max(enhanced_features))
                    }
                }
                
                print(f"Using feature vector of shape: {enhanced_features.shape}")
                print("Feature statistics:")
                for key, value in self.pipeline_data['feature_metadata']['feature_stats'].items():
                    print(f"  {key}: {value:.4f}")
                
                self.pipeline_data['features'] = enhanced_features
                return enhanced_features
            else:
                print(f"Feature dimension mismatch: base={len(base_features)}, analysis={len(analysis_features)}")
                print("Falling back to base features only")
                return base_features
                
        except Exception as e:
            print(f"Error in enhanced feature creation: {e}")
            print("Falling back to base feature generation")
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
        """Process and combine predictions to get exactly 20 numbers"""
        print("\nPost-processing predictions...")
        try:
            if predictions[0] is None or predictions[1] is None:
                raise ValueError("Invalid predictions received")
                
            prob_pred, pattern_pred = predictions
            
            # Validate prediction arrays
            if len(prob_pred) != self.num_classes or len(pattern_pred) != self.num_classes:
                raise ValueError(f"Prediction arrays must have length {self.num_classes}")
                
            # Combine predictions with weights and normalization
            prob_pred = prob_pred / np.sum(prob_pred)  # Normalize probabilities
            pattern_pred = pattern_pred / np.sum(pattern_pred)  # Normalize pattern predictions
            combined_pred = 0.4 * prob_pred + 0.6 * pattern_pred
            
            # Get top numbers and ensure uniqueness
            top_indices = np.argsort(combined_pred)[::-1]  # Sort in descending order
            final_numbers = []
            i = 0
            
            while len(final_numbers) < self.numbers_to_draw and i < len(top_indices):
                number = int(top_indices[i]) + self.numbers_range[0]
                if number not in final_numbers:  # Ensure uniqueness
                    if 1 <= number <= 80:  # Validate range
                        final_numbers.append(number)
                i += 1
            
            # Final validation
            if len(final_numbers) != self.numbers_to_draw:
                raise ValueError(f"Could not generate {self.numbers_to_draw} unique valid numbers")
            
            # Sort final numbers
            final_numbers.sort()
            
            # Store results in pipeline data with additional metadata
            self.pipeline_data.update({
                'final_prediction': final_numbers,
                'probabilities': combined_pred,
                'prediction_metadata': {
                    'prob_weight': 0.4,
                    'pattern_weight': 0.6,
                    'top_prob_numbers': sorted([int(i) + self.numbers_range[0] for i in np.argsort(prob_pred)[-5:]]),
                    'top_pattern_numbers': sorted([int(i) + self.numbers_range[0] for i in np.argsort(pattern_pred)[-5:]]),
                    'combined_confidence': float(np.mean([combined_pred[n - self.numbers_range[0]] for n in final_numbers]))
                }
            })
            
            print("\nPrediction Summary:")
            print(f"Final Numbers: {final_numbers}")
            print(f"Average Confidence: {self.pipeline_data['prediction_metadata']['combined_confidence']:.4f}")
            
            return final_numbers, combined_pred
            
        except Exception as e:
            print(f"Error in post-processing: {e}")
            return None, None

    def save_models(self, path_prefix=None):
        """Save models with timestamp"""
        try:
            if path_prefix is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path_prefix = os.path.join(self.models_dir, f'lottery_predictor_{timestamp}')
            
            # Ensure models directory exists
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Save models
            model_files = {
                '_prob_model.pkl': self.probabilistic_model,
                '_pattern_model.pkl': self.pattern_model,
                '_scaler.pkl': self.scaler
            }
            
            # Save each model file
            for suffix, model in model_files.items():
                model_path = f'{path_prefix}{suffix}'
                joblib.dump(model, model_path)
                print(f"Saved model: {os.path.basename(model_path)}")
            
            # Update timestamp file - now in the models directory
            timestamp_file = os.path.join(self.models_dir, 'model_timestamp.txt')
            with open(timestamp_file, 'w') as f:
                f.write(timestamp)
            
            print(f"Models saved successfully in {self.models_dir}")
            return True
            
        except Exception as e:
            print(f"Error saving models: {e}")
            return False

    def load_models(self, path_prefix=None):
        """Enhanced model loading with validation"""
        try:
            if path_prefix is None:
                # First try to get path from timestamp file
                timestamp_file = os.path.join(self.models_dir, 'model_timestamp.txt')
                if os.path.exists(timestamp_file):
                    with open(timestamp_file, 'r') as f:
                        timestamp = f.read().strip()
                        path_prefix = os.path.join(self.models_dir, f'lottery_predictor_{timestamp}')
                else:
                    # Fallback to finding latest model file
                    model_files = glob.glob(os.path.join(self.models_dir, "*_prob_model.pkl"))
                    if not model_files:
                        raise FileNotFoundError("No models found in directory")
                    path_prefix = max(model_files, key=os.path.getctime).replace('_prob_model.pkl', '')
            
            # Validate all required files exist
            required_files = ['_prob_model.pkl', '_pattern_model.pkl', '_scaler.pkl']
            missing_files = []
            for file in required_files:
                if not os.path.exists(f"{path_prefix}{file}"):
                    missing_files.append(file)
            
            if missing_files:
                raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")
            
            # Load models
            print(f"Loading models from: {path_prefix}")
            self.probabilistic_model = joblib.load(f'{path_prefix}_prob_model.pkl')
            self.pattern_model = joblib.load(f'{path_prefix}_pattern_model.pkl')
            self.scaler = joblib.load(f'{path_prefix}_scaler.pkl')
            
            # Update status
            self.training_status.update({
                'model_loaded': True,
                'timestamp': datetime.fromtimestamp(
                    os.path.getctime(f'{path_prefix}_prob_model.pkl')
                ),
                'features': getattr(self.probabilistic_model, 'feature_names_in_', None)
            })
            
            print("Models loaded successfully")
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

    def train_models(self, features, labels):
        """Train both models with validation for multi-label prediction"""
        try:
            print("\nStarting model training...")
            if features is None or labels is None:
                raise ValueError("Features or labels are None")
                
            if len(features) == 0 or len(labels) == 0:
                raise ValueError("Empty features or labels")
            
            # Ensure features have correct dimension (84)
            if features.shape[1] != 84:
                raise ValueError(f"Expected 84 features, got {features.shape[1]}")
                
            # Convert labels to integers and validate
            labels = np.array(labels, dtype=int)
            if not np.all((labels >= 1) & (labels <= 80)):
                raise ValueError("Labels must be between 1 and 80")
                
            # Create multi-label format
            multi_labels = np.zeros((len(labels), self.num_classes))
            for i, label in enumerate(labels):
                if isinstance(label, (list, np.ndarray)):
                    for num in label:
                        if 1 <= num <= self.num_classes:
                            multi_labels[i, num-1] = 1
                else:
                    if 1 <= label <= self.num_classes:
                        multi_labels[i, label-1] = 1
            
            # Validate label distribution
            labels_per_sample = np.sum(multi_labels, axis=1)
            print(f"\nLabel distribution:")
            print(f"- Average labels per sample: {np.mean(labels_per_sample):.2f}")
            print(f"- Min labels per sample: {np.min(labels_per_sample)}")
            print(f"- Max labels per sample: {np.max(labels_per_sample)}")
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                features, multi_labels, 
                test_size=0.2, 
                random_state=42,
                shuffle=True
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train and evaluate probabilistic model
            print("\nTraining probabilistic model...")
            self.probabilistic_model.fit(X_train_scaled, y_train)
            prob_predictions = self.probabilistic_model.predict(X_test_scaled)
            prob_score = np.mean([
                np.sum(np.sort(pred)[-self.numbers_to_draw:] == 1) / self.numbers_to_draw
                for pred in prob_predictions
            ])
            
            # Train and evaluate pattern model
            print("\nTraining pattern model...")
            self.pattern_model.fit(X_train_scaled, y_train)
            pattern_predictions = self.pattern_model.predict(X_test_scaled)
            pattern_score = np.mean([
                np.sum(np.sort(pred)[-self.numbers_to_draw:] == 1) / self.numbers_to_draw
                for pred in pattern_predictions
            ])
            
            # Update training status with enhanced metadata
            self.training_status.update({
                'success': True,
                'model_loaded': True,
                'timestamp': datetime.now(),
                'prob_score': prob_score,
                'pattern_score': pattern_score,
                'feature_dimension': features.shape[1],
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'label_stats': {
                    'mean_labels': float(np.mean(labels_per_sample)),
                    'min_labels': int(np.min(labels_per_sample)),
                    'max_labels': int(np.max(labels_per_sample))
                }
            })
            
            print(f"\nModel Training Results:")
            print(f"- Total samples: {len(features)}")
            print(f"- Training samples: {len(X_train)}")
            print(f"- Test samples: {len(X_test)}")
            print(f"- Features: {features.shape[1]}")
            print(f"- Probabilistic Model Score: {prob_score:.4f}")
            print(f"- Pattern Model Score: {pattern_score:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error training models: {e}")
            self.training_status.update({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
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
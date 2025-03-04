import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import joblib
from collections import OrderedDict
from data_analysis import DataAnalysis

class LotteryPredictor:
    def __init__(self, numbers_range=(1, 80), numbers_to_draw=20):
        self.numbers_range = numbers_range
        self.numbers_to_draw = numbers_to_draw
        self.scaler = StandardScaler()
        self.probabilistic_model = GaussianNB()
        self.pattern_model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            max_iter=500
        )
        self.analyzer = DataAnalysis([])
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

    def prepare_data(self, historical_data):
        historical_data = historical_data.sort_values('date')
        features = []
        labels = []
        
        for i in range(len(historical_data) - 5):
            window = historical_data.iloc[i:i+5]
            next_draw = historical_data.iloc[i+5]
            feature_vector = self._create_feature_vector(window)
            features.append(feature_vector)
            labels.append(next_draw['number1'])
            print(f"Processing next_draw: {next_draw[['number1', 'number2', 'number3', 'number4', 'number5', 'number6', 'number7', 'number8', 'number9', 'number10', 'number11', 'number12', 'number13', 'number14', 'number15', 'number16', 'number17', 'number18', 'number19', 'number20']]}")
        
        print(f"Features shape: {np.array(features).shape}")
        print(f"Labels shape: {np.array(labels).shape}")
        return np.array(features), np.array(labels)
    
    def _create_feature_vector(self, window):
        features = []
        number_counts = np.zeros(self.numbers_range[1] - self.numbers_range[0] + 1)
        for _, row in window.iterrows():
            for num in row[['number1', 'number2', 'number3', 'number4', 'number5', 'number6', 'number7', 'number8', 'number9', 'number10', 'number11', 'number12', 'number13', 'number14', 'number15', 'number16', 'number17', 'number18', 'number19', 'number20']]:
                number_counts[int(num) - self.numbers_range[0]] += 1
        features.extend(number_counts / len(window))
        print(f"Window features: {features}")
        
        last_draw = window.iloc[-1][['number1', 'number2', 'number3', 'number4', 'number5', 'number6', 'number7', 'number8', 'number9', 'number10', 'number11', 'number12', 'number13', 'number14', 'number15', 'number16', 'number17', 'number18', 'number19', 'number20']]
        features.extend([
            np.mean(last_draw),
            np.std(last_draw),
            len(set(last_draw) & set(range(1, 41))),
            len(set(last_draw) & set(range(41, 81)))
        ])
        print(f"Feature vector: {features}")
        return np.array(features)
    
    def _create_analysis_features(self, data):
        """Create features from data analysis"""
        print("\nGenerating analysis features...")
        
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
        
        return np.array(analysis_features)

    def _prepare_pipeline_data(self, data):
        """First pipeline stage: Prepare and validate data"""
        print("\nPreparing data for prediction pipeline...")
        if isinstance(data, pd.DataFrame):
            prepared_data = data.sort_values('date')
            self.pipeline_data['prepared_data'] = prepared_data
            return prepared_data
        else:
            raise ValueError("Input data must be a pandas DataFrame")

    def _create_enhanced_features(self, data):
        """Create enhanced feature vector with analysis"""
        print("\nGenerating enhanced features...")
        
        # Get base features
        base_features = self._create_feature_vector(data)
        
        # Get analysis features
        analysis_features = self._create_analysis_features(data)
        
        # Combine features
        enhanced_features = np.concatenate([base_features, analysis_features])
        print(f"Enhanced feature vector shape: {enhanced_features.shape}")
        
        self.pipeline_data['features'] = enhanced_features
        return enhanced_features

    def _generate_model_predictions(self, features):
        """Generate predictions from both models"""
        print("\nGenerating model predictions...")
        scaled_features = self.scaler.transform([features])
        
        prob_pred = self.probabilistic_model.predict_proba(scaled_features)[0]
        pattern_pred = self.pattern_model.predict_proba(scaled_features)[0]
        
        self.pipeline_data['prob_pred'] = prob_pred
        self.pipeline_data['pattern_pred'] = pattern_pred
        return prob_pred, pattern_pred

    def _post_process_predictions(self, predictions):
        """Process and combine predictions"""
        print("\nPost-processing predictions...")
        prob_pred, pattern_pred = predictions
        combined_pred = 0.4 * prob_pred + 0.6 * pattern_pred
        
        predicted_numbers = np.argsort(combined_pred)[-self.numbers_to_draw:]
        final_numbers = sorted([int(i) + self.numbers_range[0] for i in predicted_numbers])
        
        self.pipeline_data['final_prediction'] = final_numbers
        self.pipeline_data['probabilities'] = combined_pred
        return final_numbers, combined_pred

    def train_models(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.probabilistic_model.fit(X_train_scaled, y_train)
        self.pattern_model.fit(X_train_scaled, y_train)
    
    def predict(self, recent_draws):
        """Enhanced prediction with analysis integration"""
        try:
            # Run prediction pipeline
            result = recent_draws
            for stage_name, stage_func in self.pipeline_stages.items():
                print(f"\nExecuting pipeline stage: {stage_name}")
                result = stage_func(result)
            
            # Get final predictions
            final_numbers = self.pipeline_data['final_prediction']
            probabilities = self.pipeline_data['probabilities']
            
            # Add analysis context
            analysis_context = {
                'hot_numbers': dict(self.analyzer.hot_and_cold_numbers()[0]),
                'range_distribution': self.analyzer.number_range_analysis(),
                'common_pairs': dict(self.analyzer.find_common_pairs()),
            }
            
            print("\nPrediction Analysis Context:")
            print(f"Hot Numbers: {analysis_context['hot_numbers']}")
            print(f"Range Distribution: {analysis_context['range_distribution']}")
            print(f"Common Pairs: {analysis_context['common_pairs']}")
            
            return final_numbers, probabilities, analysis_context
            
        except Exception as e:
            print(f"Error in prediction pipeline: {str(e)}")
            # Fallback to original prediction method without analysis
            feature_vector = self._create_feature_vector(recent_draws)
            feature_vector = self.scaler.transform([feature_vector])
            
            prob_pred = self.probabilistic_model.predict_proba(feature_vector)[0]
            pattern_pred = self.pattern_model.predict_proba(feature_vector)[0]
            
            combined_pred = 0.4 * prob_pred + 0.6 * pattern_pred
            predicted_numbers = np.argsort(combined_pred)[-self.numbers_to_draw:]
            final_numbers = sorted([int(i) + self.numbers_range[0] for i in predicted_numbers])
            
            return final_numbers, combined_pred
    
    def save_models(self, path_prefix):
        joblib.dump(self.probabilistic_model, f'{path_prefix}_prob_model.pkl')
        joblib.dump(self.pattern_model, f'{path_prefix}_pattern_model.pkl')
        joblib.dump(self.scaler, f'{path_prefix}_scaler.pkl')
    
    def load_models(self, path_prefix):
        self.probabilistic_model = joblib.load(f'{path_prefix}_prob_model.pkl')
        self.pattern_model = joblib.load(f'{path_prefix}_pattern_model.pkl')
        self.scaler = joblib.load(f'{path_prefix}_scaler.pkl')

if __name__ == "__main__":
    predictor = LotteryPredictor()
    historical_data = pd.read_csv('C:/Users/MihaiNita/OneDrive - Prime Batteries/Desktop/proiectnow/Versiune1.4/src/historical_draws.csv')
    X, y = predictor.prepare_data(historical_data)
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    predictor.train_models(X, y)
    predictor.save_models('src/ml_models/lottery_predictor')
    predictor.load_models('src/ml_models/lottery_predictor')
    recent_draws = historical_data.iloc[-5:]
    print("Recent draws:")
    print(recent_draws)
    
    # Test new pipeline prediction
    prediction, probabilities, analysis = predictor.predict(recent_draws)
    print("\nFinal Results:")
    print("Predicted numbers for next draw:", prediction)
    print("Probabilities of predicted numbers:", probabilities)
    print("Analysis context:", analysis)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import joblib

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
    
    def train_models(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.probabilistic_model.fit(X_train_scaled, y_train)
        self.pattern_model.fit(X_train_scaled, y_train)
    
    def predict(self, recent_draws):
        feature_vector = self._create_feature_vector(recent_draws)
        print(f"Feature vector for prediction: {feature_vector}")
        feature_vector = self.scaler.transform([feature_vector])
        print(f"Scaled feature vector: {feature_vector}")
        
        prob_pred = self.probabilistic_model.predict_proba(feature_vector)[0]
        print(f"Probabilistic model prediction: {prob_pred}")
        pattern_pred = self.pattern_model.predict_proba(feature_vector)[0]
        print(f"Pattern recognition model prediction: {pattern_pred}")
        
        combined_pred = 0.4 * prob_pred + 0.6 * pattern_pred
        print(f"Combined prediction: {combined_pred}")
        
        predicted_numbers = np.argsort(combined_pred)[-self.numbers_to_draw:]
        print(f"Predicted numbers (indices): {predicted_numbers}")
        
        final_numbers = sorted([int(i) + self.numbers_range[0] for i in predicted_numbers])
        print(f"Predicted numbers for next draw: {final_numbers}")
        
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
    prediction, probabilities = predictor.predict(recent_draws)
    print("Predicted numbers for next draw:", prediction)
    print("Probabilities of predicted numbers:", probabilities)
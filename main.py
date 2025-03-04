import sys
import os
import pandas as pd
import numpy as np
from lottery_predictor import LotteryPredictor
from datetime import datetime, timedelta
from data_collector_selenium import KinoDataCollector
from data_analysis import DataAnalysis
from draw_handler import save_draw_to_csv
from prediction_evaluator import PredictionEvaluator
import joblib

def ensure_directories():
    directories = ['src/ml_models', 'data/processed']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def check_and_train_model():
    model_timestamp_file = 'src/model_timestamp.txt'
    models_dir = 'src/ml_models'
    needs_training = False
    if not os.path.exists(model_timestamp_file):
        needs_training = True
    else:
        with open(model_timestamp_file, 'r') as f:
            timestamp = f.read().strip()
            model_path = f'{models_dir}/lottery_predictor_{timestamp}'
            if not os.path.exists(f"{model_path}_prob_model.pkl"):
                needs_training = True
    if needs_training:
        print("No trained model found. Training new model...")
        train_and_predict()

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

def extract_date_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    return df

def get_next_draw_time(current_time):
    minutes = (current_time.minute // 5 + 1) * 5
    next_draw_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)
    return next_draw_time

def save_predictions_to_csv(predicted_numbers, probabilities, next_draw_time, file_path):
    data = {
        'Timestamp': [next_draw_time.strftime('%H:%M %d-%m-%Y')],
        'Predicted_Numbers': [','.join(map(str, predicted_numbers))],
        'Probabilities': [','.join(map(str, [probabilities[num - 1] for num in predicted_numbers]))]
    }
    df = pd.DataFrame(data)
    
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

def save_top_4_numbers_to_excel(top_4_numbers, file_path):
    df = pd.DataFrame({'Top 4 Numbers': top_4_numbers})
    df.to_excel(file_path, index=False)

def evaluate_numbers(historical_data):
    """
    Evaluate numbers based on criteria other than frequency.
    For simplicity, this example assumes a dummy evaluation function.
    Replace this with your actual evaluation logic.
    """
    number_evaluation = {i: 0 for i in range(1, 81)}
    for index, row in historical_data.iterrows():
        for i in range(1, 21):
            number = row[f'number{i}']
            number_evaluation[number] += 1  # Dummy evaluation logic

    # Sort numbers by evaluation score in descending order
    sorted_numbers = sorted(number_evaluation, key=number_evaluation.get, reverse=True)
    return sorted_numbers[:4]  # Return top 4 numbers

def train_and_predict():
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
            historical_data = extract_date_features(historical_data)
            
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
            historical_data = extract_date_features(historical_data)
        
        print("Generating ML prediction for next draw...")
        recent_draws = historical_data.tail(5).copy()
        predicted_numbers, probabilities = predictor.predict(recent_draws)

        # Evaluate and add top 4 numbers based on evaluation
        top_4_numbers = evaluate_numbers(historical_data)
        
        formatted_numbers = ','.join(map(str, predicted_numbers))
        formatted_top_4_numbers = ','.join(map(str, top_4_numbers))
        next_draw_time = get_next_draw_time(datetime.now())
        print(f"Predicted numbers for the next draw at {next_draw_time.strftime('%H:%M %d-%m-%Y')}: {formatted_numbers}")
        print(f"Top 4 numbers based on evaluation: {formatted_top_4_numbers}")
        print(f"Prediction probabilities: {[probabilities[num - 1] for num in predicted_numbers if num <= len(probabilities)]}")

        predictions_file = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\versiuni_de_care_nu_ma_ating\\Versiune1.4\\data\\processed\\predictions.csv'
        save_predictions_to_csv(predicted_numbers, probabilities, next_draw_time, predictions_file)
        
        # Save top 4 numbers to Excel
        top_4_file_path = r'C:\Users\MihaiNita\OneDrive - Prime Batteries\Desktop\proiectnow\Versiune1.4\data\processed\top_4.xlsx'
        save_top_4_numbers_to_excel(top_4_numbers, top_4_file_path)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if 'historical_data' in locals():
            print("\nFirst few rows of processed data:")
            print(historical_data.head())
            print("\nColumns in data:", historical_data.columns.tolist())

def main():
    ensure_directories()
    
    collector = KinoDataCollector()
    draws = None

    while True:
        print("\n==========================")
        print("1. Show number frequency")
        print("2. Get number suggestion")
        print("3. Fetch latest draws from lotostats.ro")
        print("4. Find common pairs")
        print("5. Find consecutive numbers")
        print("6. Number range analysis")
        print("7. Find hot and cold numbers")
        print("8. Save analysis to Excel")
        print("9. Get ML prediction")
        print("10. Evaluate prediction accuracy")
        print("11. Exit")
        print("==========================\n")

        try:
            choice = input("Choose an option (1-11): ")

            if choice == '1':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                top_numbers = analysis.get_top_numbers(20)
                print("\nTop 20 most frequent numbers:", ', '.join(map(str, top_numbers)))
                sorted_top_numbers = sorted(top_numbers)
                print("Sorted from 1 to 80:", ', '.join(map(str, sorted_top_numbers)))
            
            elif choice == '2':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                suggested_numbers = analysis.suggest_numbers()
                print("\nSuggested numbers:", ', '.join(map(str, suggested_numbers)))
            
            elif choice == '3':
                draws = collector.fetch_latest_draws()
                if draws:
                    print("\nDraws collected successfully:")
                    for i, draw in enumerate(draws, 1):
                        draw_date, numbers = draw
                        print(f"Draw {i}: Date: {draw_date}, Numbers: {', '.join(map(str, numbers))}")
                        save_draw_to_csv(draw_date, numbers)
                else:
                    print("\nFailed to fetch draws")
            
            elif choice == '4':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                common_pairs = analysis.find_common_pairs()
                print("\nCommon pairs:", common_pairs)
            
            elif choice == '5':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                consecutive_numbers = analysis.find_consecutive_numbers()
                print("\nConsecutive numbers:", consecutive_numbers)
            
            elif choice == '6':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                range_analysis = analysis.number_range_analysis()
                print("\nNumber range analysis:", range_analysis)
            
            elif choice == '7':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                hot, cold = analysis.hot_and_cold_numbers()
                print("\nHot numbers:", hot)
                print("Cold numbers:", cold)
            
            elif choice == '8':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                analysis.save_to_excel("lottery_analysis.xlsx")
                print("\nAnalysis results saved to lottery_analysis.xlsx")
            
            elif choice == '9':
                check_and_train_model()
                
                print("\nGenerating ML prediction for next draw...")
                train_and_predict()
            
            elif choice == '10':
                evaluator = PredictionEvaluator()
                evaluator.evaluate_past_predictions()
            
            elif choice == '11':
                print("\nExiting program...")
                sys.exit(0)
            
            else:
                print("\nInvalid option. Please choose 1-11")

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
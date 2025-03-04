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

def perform_complete_analysis(draws):
    """Perform all analyses and save to Excel"""
    try:
        if not draws:
            collector = KinoDataCollector()
            draws = collector.fetch_latest_draws()
        
        if draws:
            analysis = DataAnalysis(draws)
            
            # Perform all analyses
            analysis_data = {
                'frequency': analysis.get_top_numbers(20),
                'suggested_numbers': analysis.suggest_numbers(),
                'common_pairs': analysis.find_common_pairs(),
                'consecutive_numbers': analysis.find_consecutive_numbers(),
                'range_analysis': analysis.number_range_analysis(),
                'hot_cold_numbers': analysis.hot_and_cold_numbers()
            }
            
            # Save to specified Excel file
            excel_path = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\versiuni_de_care_nu_ma_ating\\Versiune1.4\\data\\processed\\lottery_analysis.xlsx'
            
            # Create Excel writer
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                # Frequency Analysis
                pd.DataFrame({
                    'Top 20 Numbers': analysis_data['frequency'],
                    'Frequency Count': [analysis.count_frequency().get(num, 0) for num in analysis_data['frequency']]
                }).to_excel(writer, sheet_name='Frequency Analysis', index=False)
                
                # Suggested Numbers
                pd.DataFrame({
                    'Suggested Numbers': analysis_data['suggested_numbers']
                }).to_excel(writer, sheet_name='Suggested Numbers', index=False)
                
                # Common Pairs
                pd.DataFrame(analysis_data['common_pairs'], 
                           columns=['Pair', 'Frequency']
                ).to_excel(writer, sheet_name='Common Pairs', index=False)
                
                # Consecutive Numbers
                pd.DataFrame({
                    'Consecutive Sets': [str(x) for x in analysis_data['consecutive_numbers']]
                }).to_excel(writer, sheet_name='Consecutive Numbers', index=False)
                
                # Range Analysis
                pd.DataFrame(analysis_data['range_analysis'].items(),
                           columns=['Range', 'Count']
                ).to_excel(writer, sheet_name='Range Analysis', index=False)
                
                # Hot and Cold Numbers
                hot, cold = analysis_data['hot_cold_numbers']
                pd.DataFrame({
                    'Hot Numbers': hot,
                    'Cold Numbers': cold
                }).to_excel(writer, sheet_name='Hot Cold Analysis', index=False)
            
            print(f"\nComplete analysis saved to: {excel_path}")
            return True
            
    except Exception as e:
        print(f"\nError in complete analysis: {str(e)}")
        return False

def test_pipeline_integration():
    """Test the integrated prediction pipeline"""
    try:
        pipeline_status = {
            'data_collection': False,
            'analysis': False,
            'prediction': False,
            'evaluation': False
        }

        # 1. Data Collection
        print("\nStep 1: Collecting data...")
        collector = KinoDataCollector()
        draws = collector.fetch_latest_draws()
        if draws:
            pipeline_status['data_collection'] = True
            print("✓ Data collection successful")
            
            # Save draws to CSV
            for draw_date, numbers in draws:
                save_draw_to_csv(draw_date, numbers)

            # 2. Analysis
            print("\nStep 2: Performing analysis...")
            if perform_complete_analysis(draws):
                pipeline_status['analysis'] = True
                print("✓ Analysis complete and saved")

            # 3. ML Prediction
            print("\nStep 3: Generating prediction...")
            check_and_train_model()
            train_and_predict()
            pipeline_status['prediction'] = True
            print("✓ Prediction generated")

            # 4. Evaluation
            print("\nStep 4: Evaluating predictions...")
            evaluator = PredictionEvaluator()
            evaluator.evaluate_past_predictions()
            pipeline_status['evaluation'] = True
            print("✓ Evaluation complete")

        return pipeline_status

    except Exception as e:
        print(f"\nError in pipeline: {str(e)}")
        return pipeline_status

def main():
    ensure_directories()
    
    collector = KinoDataCollector()
    draws = None

    while True:
        print("\n==========================")
        print("3. Fetch latest draws from lotostats.ro")
        print("8. Complete Analysis & Save")
        print("9. Get ML prediction")
        print("10. Evaluate prediction accuracy")
        print("11. Run Complete Pipeline Test")
        print("12. Exit")
        print("==========================\n")

        try:
            choice = input("Choose an option (3,8-12): ")
            
            if choice == '3':
                draws = collector.fetch_latest_draws()
                if draws:
                    print("\nDraws collected successfully:")
                    for i, draw in enumerate(draws, 1):
                        draw_date, numbers = draw
                        print(f"Draw {i}: Date: {draw_date}, Numbers: {', '.join(map(str, numbers))}")
                        save_draw_to_csv(draw_date, numbers)
                else:
                    print("\nFailed to fetch draws")
            
            elif choice == '8':
                success = perform_complete_analysis(draws)
                if success:
                    print("\nComplete analysis performed and saved successfully")
                else:
                    print("\nFailed to perform complete analysis")
            
            elif choice == '9':
                check_and_train_model()
                print("\nGenerating ML prediction for next draw...")
                train_and_predict()
            
            elif choice == '10':
                evaluator = PredictionEvaluator()
                evaluator.evaluate_past_predictions()
            
            elif choice == '11':
                print("\nRunning complete pipeline...")
                print("This will execute steps 3->8->9->10 in sequence")
                confirm = input("Continue? (y/n): ")
                if confirm.lower() == 'y':
                    status = test_pipeline_integration()
                    print("\nPipeline Test Results:")
                    for step, success in status.items():
                        print(f"{step}: {'✓' if success else '✗'}")
                    
                    if all(status.values()):
                        print("\nComplete pipeline test successful!")
                    else:
                        print("\nSome pipeline steps failed. Check the results above.")
            
            elif choice == '12':
                print("\nExiting program...")
                sys.exit(0)
            else:
                print("\nInvalid option. Please choose 3,8-12")

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
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
from draw_handler import DrawHandler  # Add this import

def ensure_directories():
    directories = ['src/ml_models', 'data/processed']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def check_and_train_model():
    """Check if a trained model exists and train if needed using DrawHandler"""
    handler = DrawHandler()
    model_path = handler._get_latest_model()
    
    if not model_path:
        print("No trained model found. Training new model...")
        if handler.train_ml_models():
            print("Model training completed successfully")
        else:
            print("Warning: Model training may have encountered issues")
    else:
        # Verify that all required model files exist
        model_files = [f"{model_path}_prob_model.pkl", f"{model_path}_pattern_model.pkl", f"{model_path}_scaler.pkl"]
        if all(os.path.exists(file) for file in model_files):
            print(f"Model found: {os.path.basename(model_path)}")
        else:
            print("Model files incomplete. Retraining model...")
            handler.train_ml_models()

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
        handler = DrawHandler()  # Use DrawHandler instead of direct LotteryPredictor
        
        # First ensure models are trained
        print("Checking/Training models...")
        if handler.train_ml_models():
            print("Models ready")
        
        # Generate prediction using pipeline
        print("\nGenerating predictions...")
        predictions, probabilities, analysis = handler.handle_prediction_pipeline()
        
        if predictions is not None:
            formatted_numbers = ','.join(map(str, predictions))
            next_draw_time = get_next_draw_time(datetime.now())
            print(f"Predicted numbers for the next draw at {next_draw_time.strftime('%H:%M %d-%m-%Y')}: {formatted_numbers}")
            print(f"Prediction probabilities: {[probabilities[num - 1] for num in predictions if num <= len(probabilities)]}")

            # Save predictions
            predictions_file = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\versiuni_de_care_nu_ma_ating\\Versiune1.4\\data\\processed\\predictions.csv'
            handler.save_predictions_to_csv(predictions, probabilities, next_draw_time.strftime('%Y-%m-%d %H:%M:%S'), predictions_file)
            
            # Optionally add top 4 numbers if still needed
            if analysis and 'hot_numbers' in analysis:
                top_4_numbers = analysis['hot_numbers'][:4]
                top_4_file_path = r'C:\Users\MihaiNita\OneDrive - Prime Batteries\Desktop\proiectnow\Versiune1.4\data\processed\top_4.xlsx'
                save_top_4_numbers_to_excel(top_4_numbers, top_4_file_path)
                print(f"Top 4 numbers based on analysis: {','.join(map(str, top_4_numbers))}")
            
            return predictions, probabilities, analysis
        else:
            print("Failed to generate predictions")
            return None, None, None
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None, None, None

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
        handler = DrawHandler()
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
                handler.save_draw_to_csv(draw_date, numbers)

            # 2. Analysis
            print("\nStep 2: Performing analysis...")
            if perform_complete_analysis(draws):
                pipeline_status['analysis'] = True
                print("✓ Analysis complete and saved")

            # 3. ML Prediction
            print("\nStep 3: Generating prediction...")
            predictions, probabilities, analysis = handler.handle_prediction_pipeline()
            if predictions is not None:
                pipeline_status['prediction'] = True
                print("✓ Prediction generated")
                
                # Display prediction results
                formatted_numbers = ','.join(map(str, predictions))
                print(f"Predicted numbers: {formatted_numbers}")
                if analysis and 'hot_numbers' in analysis:
                    print(f"Hot numbers: {analysis['hot_numbers'][:10]}")

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
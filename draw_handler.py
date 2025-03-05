import pandas as pd
from datetime import datetime, timedelta  # Added timedelta
import os
import numpy as np
from openpyxl import Workbook, load_workbook
from lottery_predictor import LotteryPredictor
from collections import Counter
from sklearn.cluster import KMeans
from data_analysis import DataAnalysis
from config.paths import PATHS, ensure_directories
from data_collector_selenium import KinoDataCollector
from prediction_evaluator import PredictionEvaluator
import joblib

class DrawHandler:
    def __init__(self):
        # Initialize paths using config
        ensure_directories()
        self.csv_file = PATHS['HISTORICAL_DATA']
        self.models_dir = PATHS['MODELS_DIR']
        self.predictions_dir = os.path.dirname(PATHS['PREDICTIONS'])
        
        # Initialize number columns
        self.number_cols = [f'number{i}' for i in range(1, 21)]
        
        # Pipeline status tracking remains the same
        self.pipeline_status = {
            'success': False,
            'stage': None,
            'error': None,
            'timestamp': None
        }
        
        # Initialize predictor
        self.predictor = LotteryPredictor()

    def handle_prediction_pipeline(self, historical_data=None):
        """Coordinates the prediction pipeline process"""
        try:
            self.pipeline_status['timestamp'] = datetime.now()
            
            if historical_data is None:
                historical_data = self._load_historical_data()
            
            # 1. Data Preparation Stage
            self.pipeline_status['stage'] = 'data_preparation'
            processed_data = self._prepare_pipeline_data(historical_data)
            
            # 2. Analysis Stage
            self.pipeline_status['stage'] = 'analysis'
            analysis_results = {}  # Initialize empty analysis results
            
            # 3. Prediction Stage
            self.pipeline_status['stage'] = 'prediction'
            model_path = self._get_latest_model()
            if model_path:
                model_files = [
                    f"{model_path}_prob_model.pkl",
                    f"{model_path}_pattern_model.pkl",
                    f"{model_path}_scaler.pkl"
                ]
                if all(os.path.exists(file) for file in model_files):
                    print(f"✓ Model found: {os.path.basename(model_path)}")
                    predictions, probabilities, analysis_results = self._run_prediction(processed_data)
                    if predictions is not None:
                        self.pipeline_status['success'] = True
                        return predictions, probabilities, analysis_results
                else:
                    print("Model files incomplete. Attempting retraining...")
                    if self.train_ml_models():
                        # After training, run prediction again with the new model
                        return self._run_prediction(processed_data)
            else:
                print("No model found. Attempting training...")
                if self.train_ml_models():
                    # After training, run prediction with the new model
                    return self._run_prediction(processed_data)
            
            # If we reach here, prediction failed
            self.pipeline_status['error'] = "Failed to generate predictions"
            return None, None, None
                
        except Exception as e:
            print(f"Error in prediction pipeline: {e}")
            self.pipeline_status['error'] = str(e)
            self.pipeline_status['success'] = False
            return None, None, None

    def train_ml_models(self, force_retrain=False):
        """Train or retrain ML models"""
        try:
            self.pipeline_status['stage'] = 'model_training'
            
            # Load historical data
            historical_data = self._load_historical_data()
            if historical_data is None:
                raise ValueError("No historical data available for training")

            # Train models using predictor
            training_success = self.predictor.train_models(historical_data, force_retrain)
            
            if training_success:
                self.pipeline_status['success'] = True
                print("Models trained successfully")
                return True
            else:
                raise Exception("Model training failed")
                
        except Exception as e:
            self.pipeline_status['error'] = str(e)
            print(f"Error in model training: {e}")
            return False

    def save_draw_to_csv(self, draw_date, draw_numbers, csv_file=None):
        if csv_file is None:
            csv_file = self.csv_file
        return save_draw_to_csv(draw_date, draw_numbers, csv_file)

    def save_predictions_to_csv(self, predicted_numbers, probabilities, timestamp, csv_file=None):
        if csv_file is None:
            csv_file = PATHS['PREDICTIONS']
        return save_predictions_to_csv(predicted_numbers, probabilities, timestamp, csv_file)

    def save_predictions_to_excel(self, predictions, probabilities, timestamp, excel_file=None):
        if excel_file is None:
            excel_file = PATHS['ANALYSIS']
        return save_predictions_to_excel(predictions, probabilities, timestamp, excel_file)
    
    def _get_latest_model(self):
        """Get the path to the latest model"""
        if os.path.exists(os.path.join(os.path.dirname(self.models_dir), 'model_timestamp.txt')):
            try:
                with open(os.path.join(os.path.dirname(self.models_dir), 'model_timestamp.txt'), 'r') as f:
                    timestamp = f.read().strip()
                    return os.path.join(self.models_dir, f'lottery_predictor_{timestamp}')
            except Exception:
                pass
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_prob_model.pkl')]
        if model_files:
            latest = max(model_files)
            return os.path.join(self.models_dir, latest.replace('_prob_model.pkl', ''))
        return None

    def _load_historical_data(self):
        """Load historical data from CSV"""
        try:
            return load_data(self.csv_file)
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return None

    def _prepare_pipeline_data(self, data):
        """Prepare data for prediction pipeline"""
        try:
            if data is None:
                return None
            
            # Basic data cleaning
            data = data.copy()
            data = data.dropna(subset=['date'])
            
            # Extract date features
            data = extract_date_features(data)
            
            # Ensure all number columns exist
            for col in self.number_cols:
                if col not in data.columns:
                    data[col] = 0
                    
            return data
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None

    def _run_prediction(self, data):
        """Run prediction with analysis integration"""
        try:
            analysis_results = {}  # Initialize empty analysis results
            
            predictor = LotteryPredictor(numbers_range=(1, 80), numbers_to_draw=20)
            model_base = self._get_latest_model()
            
            if model_base:
                # Try loading models, if fails attempt training
                if not predictor.load_models(model_base):
                    print("Model loading failed, attempting to train new model...")
                    if not self.train_ml_models():
                        raise ValueError("Could not load or train models")
                    model_base = self._get_latest_model()
                    if not predictor.load_models(model_base):
                        raise ValueError("Model loading failed after training")
                
                number_cols = [f'number{i}' for i in range(1, 21)]
                recent_draws = data.tail(5)[number_cols + ['date', 'day_of_week', 'month', 'day_of_year', 'days_since_first_draw']]
                
                # Get all three return values from predict
                predicted_numbers, probabilities, analysis = predictor.predict(recent_draws)
                
                # Merge analysis_results with the new analysis
                if analysis:
                    analysis_results.update(analysis)
                    
                    # Add model performance metrics if available
                    if predictor.training_status.get('prob_score') is not None:
                        analysis_results['model_performance'] = {
                            'probabilistic_model_score': predictor.training_status['prob_score'],
                            'pattern_model_score': predictor.training_status['pattern_score']
                        }
                
                return predicted_numbers, probabilities, analysis_results
                
            raise ValueError("No valid model base found")
            
        except Exception as e:
            print(f"Error in prediction run: {e}")
            return None, None, None

def save_draw_to_csv(draw_date, draw_numbers, csv_file=None):
    """Save draw results to CSV"""
    if csv_file is None:
        csv_file = PATHS['HISTORICAL_DATA']
    try:
        # Prepare data
        draw_data = {
            'date': [draw_date],
            **{f'number{i+1}': [num] for i, num in enumerate(sorted(draw_numbers))}
        }
        df = pd.DataFrame(draw_data)
        
        # Save to CSV
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)
        return True
    except Exception as e:
        print(f"Error saving draw to CSV: {e}")
        return False

def save_predictions_to_csv(predicted_numbers, probabilities, timestamp, csv_file=None):
    """Save predictions to CSV"""
    if csv_file is None:
        csv_file = PATHS['PREDICTIONS']
    try:
        data = {
            'Timestamp': [timestamp],
            'Predicted_Numbers': [','.join(map(str, predicted_numbers))],
            'Probabilities': [','.join(map(str, [probabilities[num - 1] for num in predicted_numbers]))]
        }
        df = pd.DataFrame(data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)
        return True
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")
        return False

def save_predictions_to_excel(predictions, probabilities, timestamp, excel_file=None):
    """Save predictions to Excel"""
    if excel_file is None:
        excel_file = PATHS['ANALYSIS']
    try:
        data = {
            'Timestamp': [timestamp],
            'Predicted_Numbers': [','.join(map(str, predictions))],
            'Probabilities': [','.join(map(str, [probabilities[num - 1] for num in predictions]))]
        }
        df = pd.DataFrame(data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(excel_file), exist_ok=True)
        
        if os.path.exists(excel_file):
            # Load existing workbook
            book = load_workbook(excel_file)
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                writer.book = book
                writer.sheets = {ws.title: ws for ws in book.worksheets}
                df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row + 1)
        else:
            # Create new workbook
            df.to_excel(excel_file, index=False)
        return True
    except Exception as e:
        print(f"Error saving predictions to Excel: {e}")
        return False

def load_data(file_path=None):
    """Load and preprocess data from CSV"""
    if file_path is None:
        file_path = PATHS['HISTORICAL_DATA']
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found")
        
    df = pd.read_csv(file_path)
    number_cols = [f'number{i}' for i in range(1, 21)]  # Define number_cols here
    
    try:
        df['date'] = pd.to_datetime(df['date'], format='%H:%M %d-%m-%Y', errors='coerce')
        df.loc[df['date'].isna(), 'date'] = pd.to_datetime(df.loc[df['date'].isna(), 'date'], errors='coerce')
    except Exception as e:
        print(f"Warning: Date conversion issue: {e}")
        
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

def save_top_4_numbers_to_excel(top_4_numbers, file_path=None):
    if file_path is None:
        file_path = os.path.join(os.path.dirname(PATHS['PREDICTIONS']), 'top_4.xlsx')
    df = pd.DataFrame({'Top 4 Numbers': top_4_numbers})
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_excel(file_path, index=False)

def evaluate_numbers(historical_data):
    """Evaluate numbers based on criteria other than frequency.
    For simplicity, this example assumes a dummy evaluation function.
    Replace this with your actual evaluation logic.
    """
    number_evaluation = {i: 0 for i in range(1, 81)}
    for index, row in historical_data.iterrows():
        for i in range(1, 21):
            number = row[f'number{i}']
            number_evaluation[number] += 1  # Dummy evaluation logic
    sorted_numbers = sorted(number_evaluation, key=number_evaluation.get, reverse=True)
    return sorted_numbers[:4]  # Return top 4 numbers

def train_and_predict():
    try:
        handler = DrawHandler()    
        
        # First ensure models are trained
        print("Checking/Training models...")
        if not handler.train_ml_models():
            raise Exception("Model training failed")
        print("Models ready")
        
        # Generate prediction using pipeline
        print("\nGenerating predictions...")
        predictions, probabilities, analysis = handler.handle_prediction_pipeline()
        
        if predictions is not None:
            # Format and display predictions
            formatted_numbers = ','.join(map(str, sorted(predictions)))
            next_draw_time = get_next_draw_time(datetime.now())
            print(f"\nPredicted numbers for next draw at {next_draw_time.strftime('%H:%M %d-%m-%Y')}:")
            print(f"Numbers: {formatted_numbers}")
            
            # Display probabilities in a readable format
            print("\nProbabilities for each predicted number:")
            for num, prob in zip(sorted(predictions), 
                               [probabilities[num - 1] for num in predictions]):
                print(f"Number {num}: {prob:.4f}")

            # Save predictions
            handler.save_predictions_to_csv(predictions, probabilities, 
                                         next_draw_time.strftime('%Y-%m-%d %H:%M:%S'))
            
            # Handle top 4 numbers if available
            if analysis and 'hot_numbers' in analysis:
                top_4_numbers = analysis['hot_numbers'][:4]
                top_4_file_path = os.path.join(os.path.dirname(PATHS['PREDICTIONS']), 'top_4.xlsx')
                save_top_4_numbers_to_excel(top_4_numbers, top_4_file_path)
                print(f"\nTop 4 numbers based on analysis: {','.join(map(str, top_4_numbers))}")
            
            return predictions, probabilities, analysis
        else:
            print("\nFailed to generate predictions")
            return None, None, None
            
    except Exception as e:
        print(f"\nError in prediction process: {str(e)}")
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
            
            # Save to Excel file using config path
            excel_path = PATHS['ANALYSIS']
            os.makedirs(os.path.dirname(excel_path), exist_ok=True)
            
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
    handler = DrawHandler()
    
    try:
        # First ensure models are trained
        print("Checking/Training models...")
        if handler.train_ml_models():
            print("Models ready")
        
        # Generate prediction using pipeline
        print("\nGenerating predictions...")
        predictions, probabilities, analysis = handler.handle_prediction_pipeline()
        
        if predictions is not None:
            print("\n=== Prediction Results ===")
            print(f"Predicted Numbers: {sorted(predictions)}")
            
            # Display probabilities for predicted numbers
            print("\nProbabilities for predicted numbers:")
            for num, prob in zip(sorted(predictions), 
                               [probabilities[num-1] for num in predictions]):
                print(f"Number {num}: {prob:.4f}")
            
            if analysis:
                print("\n=== Analysis Results ===")
                for key, value in analysis.items():
                    if key != 'clusters':  # Skip clusters for cleaner output
                        print(f"\n{key.replace('_', ' ').title()}:")
                        print(value)
                        
    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        print(f"Pipeline Status: {handler.pipeline_status}")

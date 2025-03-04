import pandas as pd
from datetime import datetime
import os
import numpy as np
from openpyxl import Workbook, load_workbook
from lottery_predictor import LotteryPredictor
from collections import Counter
from sklearn.cluster import KMeans
from data_analysis import DataAnalysis  # Add this import

class DrawHandler:
    def __init__(self):
        self.base_path = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\proiectnow\\Versiune1.4'
        self.csv_file = os.path.join(self.base_path, 'src\\historical_draws.csv')
        self.models_dir = 'src/ml_models'
        self.predictions_dir = 'data/processed'
        self.pipeline_status = {
            'success': False,
            'stage': None,
            'error': None,
            'timestamp': None
        }

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
            analysis_results = self._get_analysis_results(processed_data)
            
            # 3. Prediction Stage
            self.pipeline_status['stage'] = 'prediction'
            predictions, probabilities = self._run_prediction(processed_data, analysis_results)
            
            # 4. Results Handling
            if predictions is not None and probabilities is not None:
                self.pipeline_status['stage'] = 'results_handling'
                self._save_pipeline_results(predictions, probabilities, analysis_results)
                self.pipeline_status['success'] = True
            else:
                raise ValueError("Prediction generation failed")
                
            return predictions, probabilities, analysis_results
            
        except Exception as e:
            self.pipeline_status['success'] = False
            self.pipeline_status['error'] = str(e)
            print(f"Pipeline error in {self.pipeline_status['stage']}: {str(e)}")
            return None, None, None

    def _prepare_pipeline_data(self, data):
        """Prepares data for the prediction pipeline"""
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data['day_of_week'] = data['date'].dt.dayofweek
        data['month'] = data['date'].dt.month
        data['day_of_year'] = data['date'].dt.dayofyear
        data['days_since_first_draw'] = (data['date'] - data['date'].min()).dt.days
        return data

    def _get_analysis_results(self, data):
        """Generate analysis results for prediction enhancement"""
        analyzer = DataAnalysis(data)
        return {
            'frequency': analyzer.count_frequency(),
            'hot_numbers': analyzer.hot_and_cold_numbers()[0],
            'common_pairs': analyzer.find_common_pairs(),
            'range_analysis': analyzer.number_range_analysis()
        }

    def _run_prediction(self, data, analysis_results):
        """Run prediction with analysis integration"""
        try:
            predictor = LotteryPredictor(numbers_range=(1, 80), numbers_to_draw=20)
            model_base = self._get_latest_model()
            
            if model_base:
                predictor.load_models(model_base)
                number_cols = [f'number{i}' for i in range(1, 21)]
                recent_draws = data.tail(5)[number_cols + ['date', 'day_of_week', 'month', 'day_of_year', 'days_since_first_draw']]
                
                # Now get all three return values from predict
                predictions, probabilities, analysis = predictor.predict(recent_draws)
                
                # Merge analysis_results with the new analysis
                if analysis:
                    analysis_results.update(analysis)
                    
                return predictions, probabilities
                
            return None, None
            
        except Exception as e:
            print(f"Error in prediction run: {e}")
            return None, None

    # Keep all existing functions unchanged
    def save_draw_to_csv(self, draw_date, draw_numbers, csv_file=None):
        if csv_file is None:
            csv_file = self.csv_file
        return save_draw_to_csv(draw_date, draw_numbers, csv_file)

    def save_predictions_to_csv(self, predicted_numbers, probabilities, timestamp, csv_file=None):
        if csv_file is None:
            csv_file = os.path.join(self.predictions_dir, 'predictions.csv')
        return save_predictions_to_csv(predicted_numbers, probabilities, timestamp, csv_file)

    def save_predictions_to_excel(self, predictions, probabilities, timestamp, excel_file=None):
        if excel_file is None:
            excel_file = os.path.join(self.predictions_dir, 'predictions.xlsx')
        return save_predictions_to_excel(predictions, probabilities, timestamp, excel_file)

    def train_ml_models(self, csv_file=None, models_dir=None):
        """Train ML models using LotteryPredictor"""
        if csv_file is None:
            csv_file = self.csv_file
        if models_dir is None:
            models_dir = self.models_dir
            
        try:
            predictor = LotteryPredictor(numbers_range=(1, 80), numbers_to_draw=20)
            predictor.models_dir = models_dir
            
            # Load and prepare data
            historical_data = pd.read_csv(csv_file)
            processed_data = self._prepare_pipeline_data(historical_data)
            
            # Train and save models
            prediction, probabilities, analysis = predictor.train_and_predict(
                historical_data=processed_data
            )
            
            return True
        except Exception as e:
            print(f"Error training models: {e}")
            return False

    def get_ml_prediction(self, csv_file=None):
        """Get prediction using LotteryPredictor"""
        if csv_file is None:
            csv_file = self.csv_file
            
        try:
            predictor = LotteryPredictor(numbers_range=(1, 80), numbers_to_draw=20)
            historical_data = pd.read_csv(csv_file)
            
            # Prepare data
            processed_data = self._prepare_pipeline_data(historical_data)
            
            # Get prediction
            prediction, probabilities, analysis = predictor.train_and_predict(
                historical_data=processed_data
            )
            
            return prediction, probabilities, analysis
        except Exception as e:
            print(f"Error getting prediction: {e}")
            return None, None, None

    # Helper methods
    def _load_historical_data(self):
        """Load and validate historical data"""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"Historical data file not found: {self.csv_file}")
        return pd.read_csv(self.csv_file)

    def _get_latest_model(self):
        """Get the path to the latest model"""
        if os.path.exists('src/model_timestamp.txt'):
            try:
                with open('src/model_timestamp.txt', 'r') as f:
                    timestamp = f.read().strip()
                    return f'{self.models_dir}/lottery_predictor_{timestamp}'
            except Exception:
                pass
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_prob_model.pkl')]
        if model_files:
            latest = max(model_files)
            return os.path.join(self.models_dir, latest.replace('_prob_model.pkl', ''))
        return None

    def _save_pipeline_results(self, predictions, probabilities, analysis_results):
        """Save pipeline results including analysis"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save predictions
        if predictions is not None and probabilities is not None:
            self.save_predictions_to_csv(predictions, probabilities, timestamp)
            self.save_predictions_to_excel(predictions, probabilities, timestamp)
        
        # Save analysis results
        if analysis_results:
            analysis_file = os.path.join(self.predictions_dir, 'analysis_results.csv')
            analysis_df = pd.DataFrame([{
                'timestamp': timestamp,
                'hot_numbers': str(analysis_results.get('hot_numbers', [])),
                'frequency_data': str(analysis_results.get('frequency', {})),
                'range_analysis': str(analysis_results.get('range_analysis', {}))
            }])
            
            if os.path.exists(analysis_file):
                analysis_df.to_csv(analysis_file, mode='a', header=False, index=False)
            else:
                analysis_df.to_csv(analysis_file, index=False)

# Keep original functions as module-level functions for backward compatibility
def save_draw_to_csv(draw_date, draw_numbers, csv_file='C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\proiectnow\\Versiune1.4\\src\\historical_draws.csv'):
    """Save draw results to CSV"""
    try:
        # Prepare data
        draw_data = {
            'date': [draw_date],
            **{f'number{i+1}': [num] for i, num in enumerate(sorted(draw_numbers))}
        }
        df = pd.DataFrame(draw_data)
        
        # Save to CSV
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)
        return True
    except Exception as e:
        print(f"Error saving draw to CSV: {e}")
        return False

def save_predictions_to_csv(predicted_numbers, probabilities, timestamp, csv_file='data/processed/predictions.csv'):
    """Save predictions to CSV"""
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

def save_predictions_to_excel(predictions, probabilities, timestamp, excel_file='data/processed/predictions.xlsx'):
    """Save predictions to Excel"""
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

def train_ml_models(csv_file='C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\proiectnow\\Versiune1.4\\src\\historical_draws.csv', models_dir='src/ml_models'):
    """Original train_ml_models function"""
    # Original implementation remains unchanged
    pass  # Replace with the actual implementation of the function

def get_ml_prediction(csv_file='C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\proiectnow\\Versiune1.4\\src\\historical_draws.csv'):
    """Original get_ml_prediction function"""
    # Original implementation remains unchanged
    pass  # Replace with the actual implementation of the function

if __name__ == "__main__":
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
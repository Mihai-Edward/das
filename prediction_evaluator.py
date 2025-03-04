import pandas as pd
from datetime import datetime
import numpy as np
import os

class PredictionEvaluator:
    def __init__(self):
        self.predictions_file = 'C:\\Users\\MihaiNita\\OneDrive - Prime Batteries\\Desktop\\versiuni_de_care_nu_ma_ating\\Versiune1.4\\data\\processed\\predictions.csv'
        self.historical_file = 'src/historical_draws.csv'
        self.results_dir = 'data/processed'
        self.results_file = os.path.join(self.results_dir, 'prediction_results.xlsx')
        
        # Ensure directory exists
        os.makedirs(self.results_dir, exist_ok=True)

    def compare_prediction_with_actual(self, predicted_numbers, actual_numbers):
        """
        Compare predicted numbers with actual draw numbers
        
        Args:
            predicted_numbers (list): List of predicted numbers
            actual_numbers (list): List of actual draw numbers
            
        Returns:
            dict: Dictionary containing comparison results
        """
        predicted_set = set(predicted_numbers)
        actual_set = set(actual_numbers)
        
        correct_numbers = predicted_set.intersection(actual_set)
        accuracy = len(correct_numbers) / 20  # Since we draw 20 numbers
        
        return {
            'correct_numbers': sorted(list(correct_numbers)),
            'num_correct': len(correct_numbers),
            'accuracy': accuracy,
            'predicted_numbers': sorted(list(predicted_numbers)),
            'actual_numbers': sorted(list(actual_numbers))
        }

    def save_comparison(self, predicted_numbers, actual_numbers, draw_date=None):
        """
        Save prediction comparison to Excel
        
        Args:
            predicted_numbers (list): List of predicted numbers
            actual_numbers (list): List of actual draw numbers
            draw_date (str, optional): Date of the draw. Defaults to current datetime.
        
        Returns:
            dict: Comparison results
        """
        if draw_date is None:
            draw_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        result = self.compare_prediction_with_actual(predicted_numbers, actual_numbers)
        
        data = {
            'Date': [draw_date],
            'Predicted_Numbers': [str(result['predicted_numbers'])],
            'Actual_Numbers': [str(result['actual_numbers'])],
            'Correct_Numbers': [str(result['correct_numbers'])],
            'Number_Correct': [result['num_correct']],
            'Accuracy': [f"{result['accuracy']*100:.2f}%"]
        }
        
        df = pd.DataFrame(data)
        
        try:
            # Try to load existing file
            if os.path.exists(self.results_file):
                existing_df = pd.read_excel(self.results_file)
                df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not load existing results file: {e}")
        
        # Save to Excel
        try:
            df.to_excel(self.results_file, index=False)
            print(f"\nResults saved to {self.results_file}")
        except Exception as e:
            print(f"Warning: Could not save results: {e}")
        
        return result

    def get_performance_stats(self):
        """
        Calculate overall prediction performance statistics
        
        Returns:
            dict: Dictionary containing performance statistics
        """
        try:
            if not os.path.exists(self.results_file):
                return None
                
            df = pd.read_excel(self.results_file)
            if len(df) == 0:
                return None
                
            stats = {
                'total_predictions': len(df),
                'average_correct': df['Number_Correct'].mean(),
                'best_prediction': df['Number_Correct'].max(),
                'worst_prediction': df['Number_Correct'].min(),
                'average_accuracy': df['Number_Correct'].mean() / 20 * 100
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return None

    def display_summary_results(self, stats):
        """
        Display summary of overall performance statistics
        
        Args:
            stats (dict): Performance statistics
        """
        if stats:
            print("\n=== Overall Performance ===")
            print(f"Total predictions evaluated: {stats['total_predictions']}")
            print(f"Average correct numbers: {stats['average_correct']:.1f}")
            print(f"Best prediction: {stats['best_prediction']} correct numbers")
            print(f"Worst prediction: {stats['worst_prediction']} correct numbers")
            print(f"Average accuracy: {stats['average_accuracy']:.1f}%")
        else:
            print("\nNo performance statistics available.")

    def evaluate_past_predictions(self):
        """
        Evaluate past predictions against the actual numbers in historical draws.
        """
        try:
            # Load predictions
            predictions_df = pd.read_csv(self.predictions_file, names=["Timestamp", "Predicted_Numbers", "Probabilities"])
            if predictions_df.empty:
                print("\nNo predictions found in the file. Please use option 9 first to generate predictions.")
                return
                
            # Load actual numbers from historical draws
            if not os.path.exists(self.historical_file):
                print("\nNo historical draw data found.")
                return
                
            historical_df = pd.read_csv(self.historical_file, header=None)
            if historical_df.empty:
                print("\nNo historical draw data to compare.")
                return
                
            # Fixing the date format in the historical data
            historical_df[20] = pd.to_datetime(historical_df[20], format='%H:%M %d-%m-%Y', errors='coerce')
            
            # Iterate over past predictions and compare with actual results
            for _, row in predictions_df.iterrows():
                prediction_date = row['Timestamp']
                predicted_numbers = list(map(int, row['Predicted_Numbers'].strip('[]').split(',')))

                parsed_prediction_date = pd.to_datetime(prediction_date, errors='coerce')
                if parsed_prediction_date > datetime.now():
                    continue
                
                # Find the corresponding actual draw by date
                actual_row = historical_df[historical_df[20] == parsed_prediction_date]
                if actual_row.empty:
                    continue
                    
                actual_numbers = actual_row.iloc[0][:20].values
                actual_numbers = list(map(int, actual_numbers))

                # Compare and save results
                self.save_comparison(predicted_numbers, actual_numbers, prediction_date)
            
            # Display overall performance statistics
            stats = self.get_performance_stats()
            self.display_summary_results(stats)
                
        except Exception as e:
            print(f"\nError in evaluation: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    evaluator = PredictionEvaluator()
    evaluator.evaluate_past_predictions()
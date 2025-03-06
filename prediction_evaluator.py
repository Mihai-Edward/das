import os
import sys
from typing import Counter
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import directly from config folder
from config.paths import PATHS, ensure_directories

class PredictionEvaluator:
    def __init__(self):
        # Use PATHS configuration for relative paths
       def __init__(self):
    # Use PATHS configuration for relative paths
        self.predictions_file = PATHS['PREDICTIONS']  # Use the standardized predictions file
        self.historical_file = PATHS['HISTORICAL_DATA']
        self.results_dir = os.path.dirname(PATHS['PREDICTIONS'])
        self.results_file = PATHS['ANALYSIS']
        self.analysis_file = PATHS['ANALYSIS']
        # Ensure directories exist
        ensure_directories()
        
        # Initialize evaluation metrics
        self.evaluation_metrics = {
            'accuracy_history': [],
            'precision_history': [],
            'recall_history': [],
            'prediction_confidence': []
        }

    def compare_prediction_with_actual(self, predicted_numbers, actual_numbers):
        """
        Compare predicted numbers with actual draw numbers with enhanced metrics
        
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
        
        # Calculate additional metrics
        precision = len(correct_numbers) / len(predicted_set) if predicted_set else 0
        recall = len(correct_numbers) / len(actual_set) if actual_set else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'correct_numbers': sorted(list(correct_numbers)),
            'num_correct': len(correct_numbers),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'predicted_numbers': sorted(list(predicted_numbers)),
            'actual_numbers': sorted(list(actual_numbers))
        }

    def analyze_prediction_patterns(self, results_df):
        """
        Analyze patterns in prediction results
        
        Args:
            results_df (DataFrame): DataFrame containing prediction results
            
        Returns:
            dict: Dictionary containing pattern analysis
        """
        patterns = {
            'frequent_correct': Counter(),
            'frequent_missed': Counter(),
            'time_based_accuracy': {},
            'streak_analysis': {
                'current_streak': 0,
                'best_streak': 0
            }
        }
        
        for _, row in results_df.iterrows():
            correct_nums = eval(row['Correct_Numbers'])
            predicted = eval(row['Predicted_Numbers'])
            actual = eval(row['Actual_Numbers'])
            
            # Track frequently correct numbers
            patterns['frequent_correct'].update(correct_nums)
            
            # Track frequently missed numbers
            missed = set(actual) - set(predicted)
            patterns['frequent_missed'].update(missed)
            
            # Time-based analysis
            date = pd.to_datetime(row['Date'])
            time_key = f"{date.hour:02d}:00"
            if time_key not in patterns['time_based_accuracy']:
                patterns['time_based_accuracy'][time_key] = []
            patterns['time_based_accuracy'][time_key].append(row['Number_Correct'])
            
            # Streak analysis
            if row['Number_Correct'] >= 5:  # Consider 5+ correct as a "good" prediction
                patterns['streak_analysis']['current_streak'] += 1
                patterns['streak_analysis']['best_streak'] = max(
                    patterns['streak_analysis']['best_streak'],
                    patterns['streak_analysis']['current_streak']
                )
            else:
                patterns['streak_analysis']['current_streak'] = 0
        
        return patterns

    def save_comparison(self, predicted_numbers, actual_numbers, draw_date=None, prediction_confidence=None):
        """
        Save prediction comparison to Excel with enhanced metrics
        
        Args:
            predicted_numbers (list): List of predicted numbers
            actual_numbers (list): List of actual draw numbers
            draw_date (str, optional): Date of the draw
            prediction_confidence (float, optional): Confidence score of the prediction
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
            'Accuracy': [f"{result['accuracy']*100:.2f}%"],
            'Precision': [f"{result['precision']*100:.2f}%"],
            'Recall': [f"{result['recall']*100:.2f}%"],
            'F1_Score': [f"{result['f1_score']*100:.2f}%"]
        }
        
        if prediction_confidence is not None:
            data['Prediction_Confidence'] = [prediction_confidence]
        
        df = pd.DataFrame(data)
        
        try:
            if os.path.exists(self.results_file):
                existing_df = pd.read_excel(self.results_file)
                df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not load existing results file: {e}")
        
        try:
            df.to_excel(self.results_file, index=False)
            print(f"\nResults saved to {self.results_file}")
            
            # Update evaluation metrics
            self.evaluation_metrics['accuracy_history'].append(result['accuracy'])
            self.evaluation_metrics['precision_history'].append(result['precision'])
            self.evaluation_metrics['recall_history'].append(result['recall'])
            if prediction_confidence is not None:
                self.evaluation_metrics['prediction_confidence'].append(prediction_confidence)
            
        except Exception as e:
            print(f"Warning: Could not save results: {e}")
        
        return result

    def get_performance_stats(self):
        """Calculate enhanced performance statistics"""
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
                'average_accuracy': df['Number_Correct'].mean() / 20 * 100,
                'recent_trend': self._calculate_trend(df['Number_Correct'].tail(5).values),
                'consistency_score': np.std(df['Number_Correct']),
                'improvement_rate': self._calculate_improvement_rate(df)
            }
            
            # Add pattern analysis
            patterns = self.analyze_prediction_patterns(df)
            stats.update({
                'best_streak': patterns['streak_analysis']['best_streak'],
                'most_frequent_correct': dict(patterns['frequent_correct'].most_common(5)),
                'most_frequently_missed': dict(patterns['frequent_missed'].most_common(5))
            })
            
            return stats
            
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return None

    def _calculate_trend(self, recent_values):
        """Calculate the trend in recent predictions"""
        if len(recent_values) < 2:
            return 0
        return np.polyfit(range(len(recent_values)), recent_values, 1)[0]

    def _calculate_improvement_rate(self, df):
        """Calculate the rate of improvement over time"""
        if len(df) < 2:
            return 0
        first_half_avg = df['Number_Correct'].iloc[:len(df)//2].mean()
        second_half_avg = df['Number_Correct'].iloc[len(df)//2:].mean()
        return ((second_half_avg - first_half_avg) / first_half_avg) * 100 if first_half_avg > 0 else 0

    def plot_performance_trends(self):
        """Generate and save performance trend plots"""
        if not os.path.exists(self.results_file):
            return
            
        df = pd.read_excel(self.results_file)
        if df.empty:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Accuracy trend
        plt.subplot(2, 2, 1)
        plt.plot(df['Number_Correct'])
        plt.title('Prediction Accuracy Over Time')
        plt.xlabel('Prediction Number')
        plt.ylabel('Correct Numbers')
        
        # Distribution of correct predictions
        plt.subplot(2, 2, 2)
        sns.histplot(data=df['Number_Correct'])
        plt.title('Distribution of Correct Predictions')
        
        # Save plot
        plot_file = os.path.join(self.results_dir, 'performance_trends.png')
        plt.savefig(plot_file)
        plt.close()

    def evaluate_past_predictions(self):
        """Evaluate past predictions with enhanced analysis"""
        try:
            # Load predictions and historical data
            if not os.path.exists(self.predictions_file):
                print("\nNo predictions file found.")
                return
                
            if not os.path.exists(self.historical_file):
                print("\nNo historical draw data found.")
                return
            
            # Load predictions with proper format
            predictions_df = pd.read_csv(self.predictions_file, header=None)
            historical_df = pd.read_csv(self.historical_file, header=None)
            
            if predictions_df.empty or historical_df.empty:
                print("\nNo data to compare.")
                return
            
            # Convert dates to datetime
            current_time = datetime.now()
            predictions_df[20] = pd.to_datetime(predictions_df[20], format='%H:%M %d-%m-%Y', errors='coerce')
            historical_df[20] = pd.to_datetime(historical_df[20], format='%H:%M %d-%m-%Y', errors='coerce')
            
            # Filter out future predictions
            past_predictions = predictions_df[predictions_df[20] < current_time]
            
            evaluation_results = []
            for idx, pred_row in past_predictions.iterrows():
                pred_date = pred_row[20]
                
                # Find matching historical draw
                actual_row = historical_df[historical_df[20] == pred_date]
                if actual_row.empty:
                    continue
                
                # Get predicted and actual numbers
                predicted_numbers = pred_row[:20].astype(int).tolist()
                actual_numbers = actual_row.iloc[0][:20].astype(int).tolist()
                
                # Compare and save results
                result = self.save_comparison(
                    predicted_numbers, 
                    actual_numbers, 
                    pred_date.strftime('%Y-%m-%d %H:%M:%S')
                )
                evaluation_results.append(result)
            
            if evaluation_results:
                # Generate performance statistics
                stats = self.get_performance_stats()
                self.display_summary_results(stats)
                
                # Generate performance plots
                self.plot_performance_trends()
            else:
                print("\nNo past predictions found to evaluate.")
                
        except Exception as e:
            print(f"\nError in evaluation: {str(e)}")
            print("Please try again.")

    def display_summary_results(self, stats):
        """Display enhanced summary of performance statistics"""
        if stats:
            print("\n=== Overall Performance ===")
            print(f"Total predictions evaluated: {stats['total_predictions']}")
            print(f"Average correct numbers: {stats['average_correct']:.1f}")
            print(f"Best prediction: {stats['best_prediction']} correct numbers")
            print(f"Worst prediction: {stats['worst_prediction']} correct numbers")
            print(f"Average accuracy: {stats['average_accuracy']:.1f}%")
            print(f"Best streak: {stats['best_streak']} predictions")
            print(f"\nRecent trend: {'Improving' if stats['recent_trend'] > 0 else 'Declining'}")
            print(f"Improvement rate: {stats['improvement_rate']:.1f}%")
            
            print("\nMost frequently correct numbers:")
            for num, count in stats['most_frequent_correct'].items():
                print(f"Number {num}: {count} times")
                
            print("\nMost frequently missed numbers:")
            for num, count in stats['most_frequently_missed'].items():
                print(f"Number {num}: {count} times")
        else:
            print("\nNo performance statistics available.")

if __name__ == "__main__":
    evaluator = PredictionEvaluator()
    evaluator.evaluate_past_predictions()
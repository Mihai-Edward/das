# File: automation/operations.py
import os
import sys
import time
from datetime import datetime

# Add src directory directly to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)  # Add src directory directly
sys.path.insert(0, project_root)  # Also add project root

# Now the imports should work correctly
from config.paths import PATHS, ensure_directories
from draw_handler import DrawHandler, save_draw_to_csv, perform_complete_analysis, train_and_predict
from data_collector_selenium import KinoDataCollector
from prediction_evaluator import PredictionEvaluator

def fetch_latest_draws():
    """
    Programmatic version of menu option 3 - Fetch latest draws from website.
    
    Returns:
        tuple: (success, result)
            - success (bool): True if operation succeeded, False otherwise
            - result: If success, a list of draws; if failure, an error message
    """
    try:
        print("\n[Automation] Fetching latest draws from website...")
        collector = KinoDataCollector()
        draws = collector.fetch_latest_draws()
        
        if not draws:
            print("[Automation] No draws fetched.")
            return False, "No draws fetched"
            
        print(f"[Automation] Fetched {len(draws)} draws from website.")
        
        # Save draws to CSV
        for draw_date, numbers in draws:
            save_draw_to_csv(draw_date, numbers)
            
        print("[Automation] Draws saved to CSV.")
        return True, draws
    except Exception as e:
        error_msg = f"Error fetching latest draws: {str(e)}"
        print(f"[Automation] {error_msg}")
        return False, error_msg


def perform_analysis(draws=None):
    """
    Programmatic version of menu option 8 - Perform complete analysis.
    
    Args:
        draws: Optional list of draws to analyze. If None, will use draws from CSV.
        
    Returns:
        tuple: (success, result)
            - success (bool): True if operation succeeded, False otherwise
            - result: If success, None; if failure, an error message
    """
    try:
        print("\n[Automation] Performing complete analysis...")
        success = perform_complete_analysis(draws)
        
        if success:
            print("[Automation] Analysis completed successfully.")
            return True, None
        else:
            error_msg = "Analysis failed to complete."
            print(f"[Automation] {error_msg}")
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Error performing analysis: {str(e)}"
        print(f"[Automation] {error_msg}")
        return False, error_msg


def generate_prediction():
    """
    Programmatic version of menu option 9 - Generate ML prediction.
    
    Returns:
        tuple: (success, result)
            - success (bool): True if operation succeeded, False otherwise
            - result: If success, a tuple of (predictions, probabilities, analysis);
                     if failure, an error message
    """
    try:
        print("\n[Automation] Generating ML prediction...")
        handler = DrawHandler()
        
        # Ensure models are trained
        if not handler._get_latest_model():
            print("[Automation] No models found. Training models...")
            if not handler.train_ml_models():
                return False, "Model training failed"
                
        # Generate prediction
        predictions, probabilities, analysis = handler.handle_prediction_pipeline()
        
        if predictions is None or len(predictions) == 0:
            print("[Automation] Failed to generate predictions.")
            return False, "No predictions generated"
            
        # Save predictions
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        handler.save_predictions_to_csv(predictions, probabilities, timestamp)
        
        # Display basic info
        print(f"[Automation] Generated prediction: {sorted(predictions)}")
        
        return True, (predictions, probabilities, analysis)
        
    except Exception as e:
        error_msg = f"Error generating prediction: {str(e)}"
        print(f"[Automation] {error_msg}")
        return False, error_msg


def evaluate_prediction():
    """
    Programmatic version of menu option 10 - Evaluate prediction.
    
    Returns:
        tuple: (success, result)
            - success (bool): True if operation succeeded, False otherwise
            - result: If success, evaluation results; if failure, an error message
    """
    try:
        print("\n[Automation] Evaluating prediction...")
        evaluator = PredictionEvaluator()
        evaluator.evaluate_past_predictions()
        
        # Get performance stats
        stats = evaluator.get_performance_stats()
        
        if stats:
            print(f"[Automation] Evaluation complete. Last accuracy: {stats.get('average_accuracy', 0):.2f}%")
            return True, stats
        else:
            print("[Automation] Evaluation complete but no stats available.")
            return True, None
            
    except Exception as e:
        error_msg = f"Error evaluating prediction: {str(e)}"
        print(f"[Automation] {error_msg}")
        return False, error_msg


def test_operation():
    """
    Simple function to test importing and using the operations module.
    
    Returns:
        bool: True if the test passes, False otherwise
    """
    try:
        print("Operations module imported successfully!")
        ensure_directories()
        print("Directories checked successfully!")
        return True
    except Exception as e:
        print(f"Error testing operations: {str(e)}")
        return False


if __name__ == "__main__":
    # Simple test code for the operations
    test_operation()
    
    # Uncomment to test specific operations
    # success, result = fetch_latest_draws()
    # print(f"Fetch success: {success}")
    
    # success, result = perform_analysis()
    # print(f"Analysis success: {success}")
    
    # success, result = generate_prediction()
    # print(f"Prediction success: {success}")
    
    # success, result = evaluate_prediction()
    # print(f"Evaluation success: {success}")
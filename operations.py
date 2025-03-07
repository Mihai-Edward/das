# File: automation/operations.py
import os
import sys
import time
from datetime import datetime, timedelta
import pytz

# Add src directory directly to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)  # Add src directory directly
sys.path.insert(0, project_root)  # Also add project root

# Now the imports should work correctly
from config.paths import PATHS, ensure_directories
from src.draw_handler import DrawHandler, save_draw_to_csv, perform_complete_analysis, train_and_predict
from src.data_collector_selenium import KinoDataCollector
from src.prediction_evaluator import PredictionEvaluator
from automation.scheduler import DrawScheduler

# In automation/operations.py

def fetch_latest_draws(max_retries=3, retry_delay=10):
    """
    Mirror the working data_collector_selenium.py functionality
    """
    try:
        print("\n[Automation] Fetching latest draws from website...")
        collector = KinoDataCollector()
        
        # First sort existing historical draws
        print("[Automation] Sorting historical draws...")
        collector.sort_historical_draws()
        
        # Then fetch new draws with the proven working method
        draws = collector.fetch_latest_draws(num_draws=1)  # Get latest draw
        
        if not draws:
            print("[Automation] No draws fetched after all attempts.")
            return False, "No draws fetched"
            
        print(f"[Automation] Fetched {len(draws)} draws from website.")
        
        # Sort again after collecting new draws
        print("[Automation] Sorting updated historical draws...")
        if collector.sort_historical_draws():
            print("[Automation] Historical draws successfully sorted from newest to oldest")
        else:
            print("[Automation] Error occurred while sorting draws")
            
        print("\nCollection Status:", collector.collection_status)
        return True, draws

    except Exception as e:
        error_msg = f"Error fetching latest draws: {str(e)}"
        print(f"[Automation] {error_msg}")
        return False, error_msg

def perform_analysis(draws=None):
    """
    Perform complete analysis on draws.
    
    Args:
        draws: Optional list of draws to analyze. If None, uses draws from CSV.
        
    Returns:
        tuple: (success, result)
            - success (bool): True if operation succeeded
            - result: None if successful, error message if failed
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

def generate_prediction(for_draw_time=None):
    """
    Generate prediction for next draw.
    
    Args:
        for_draw_time: Optional datetime for target draw
    
    Returns:
        tuple: (success, result)
            - success (bool): True if operation succeeded
            - result: (predictions, probabilities, analysis) if successful,
                     error message if failed
    """
    try:
        print("\n[Automation] Generating ML prediction...")
        from src.draw_handler import DrawHandler
        handler = DrawHandler()
        predictions, probabilities, analysis = handler.handle_prediction_pipeline()

        # Set target draw time
        if for_draw_time is None:
            current_time = datetime.now(pytz.UTC)
            scheduler = DrawScheduler()
            for_draw_time = scheduler.get_next_draw_time(current_time)

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
            
        # Save predictions with UTC timestamp
        timestamp = for_draw_time.strftime('%Y-%m-%d %H:%M:%S')
        handler.save_predictions_to_csv(predictions, probabilities, timestamp)
        
        # Display basic info
        print(f"[Automation] Generated prediction: {sorted(predictions)}")
        print(f"[Automation] For draw at: {timestamp} UTC")
        
        return True, (predictions, probabilities, analysis)
        
    except Exception as e:
        error_msg = f"Error generating prediction: {str(e)}"
        print(f"[Automation] {error_msg}")
        return False, error_msg

def evaluate_prediction(draw_time=None):
    """
    Evaluate prediction accuracy.
    
    Args:
        draw_time: Optional datetime of draw to evaluate
    
    Returns:
        tuple: (success, result)
            - success (bool): True if operation succeeded
            - result: Evaluation stats if successful, error message if failed
    """
    try:
        print("\n[Automation] Evaluating prediction...")
        if draw_time:
            print(f"[Automation] Evaluating draw from: {draw_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
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
    """Test importing and basic functionality."""
    try:
        print("Operations module imported successfully!")
        ensure_directories()
        print("Directories checked successfully!")
        return True
    except Exception as e:
        print(f"Error testing operations: {str(e)}")
        return False

if __name__ == "__main__":
    # Test the operations module
    test_operation()
    
    # Optional: Test individual operations
    # success, result = fetch_latest_draws()
    # print(f"Fetch success: {success}")
    
    # success, result = perform_analysis()
    # print(f"Analysis success: {success}")
    
    # success, result = generate_prediction()
    # print(f"Prediction success: {success}")
    
    # success, result = evaluate_prediction()
    # print(f"Evaluation success: {success}")
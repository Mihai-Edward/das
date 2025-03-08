# automation/operations.py

import os
import sys
import time
from datetime import datetime, timedelta
import pytz

# Add project root to path if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules
from config.paths import PATHS, ensure_directories

# Define the standard timezone to use across all operations
TIMEZONE = pytz.timezone('Europe/Bucharest')  # UTC+2

def test_operation():
    """Test operation to verify imports and configuration."""
    print("\nTesting operations module...")
    
    # Verify paths configuration
    ensure_directories()
    
    # Test importing critical modules
    try:
        from src.draw_handler import DrawHandler
        from src.lottery_predictor import LotteryPredictor
        from src.data_collector_selenium import KinoDataCollector
        print("✓ All required modules imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        raise
        
    return True

def collect_data_operation(num_draws=10):
    """
    Collect data from website using selenium.
    
    This operation corresponds to the FETCHING state in the state machine.
    
    Args:
        num_draws: Number of draws to fetch
        
    Returns:
        tuple: (success, message_or_data)
    """
    try:
        print("\n[Automation] Fetching latest draws from website...")
        
        # Import collector after path setup
        from src.data_collector_selenium import KinoDataCollector
        
        # Initialize collector
        collector = KinoDataCollector()
        
        # First sort existing data
        print("[Automation] Sorting historical draws...")
        collector.sort_historical_draws()
        
        # Get latest draws
        start_time = datetime.now(TIMEZONE)
        draws = collector.fetch_latest_draws(num_draws=num_draws)
        fetch_duration = (datetime.now(TIMEZONE) - start_time).total_seconds()
        
        if not draws:
            return False, "Failed to fetch draws"
            
        # Sort again after adding new data
        print("[Automation] Sorting updated historical draws...")
        if collector.sort_historical_draws():
            print("[Automation] Historical draws successfully sorted from newest to oldest")
        else:
            print("[Automation] Warning: Failed to sort draws, but continuing")
            
        print(f"[Automation] Fetched {len(draws)} draws from website in {fetch_duration:.2f} seconds.")
        return True, {
            "draws": draws,
            "count": len(draws),
            "duration": fetch_duration
        }
        
    except Exception as e:
        error_msg = f"Error collecting data: {str(e)}"
        print(f"[Automation] {error_msg}")
        return False, error_msg

def analyze_data_operation():
    """
    Perform analysis on collected data.
    
    This operation corresponds to the ANALYZING state in the state machine.
    
    Returns:
        tuple: (success, message_or_data)
    """
    try:
        print("\n[Automation] Performing complete analysis...")
        start_time = datetime.now(TIMEZONE)
        
        # Import handler after path setup
        from src.draw_handler import DrawHandler
        from src.data_collector_selenium import KinoDataCollector
        
        # Get draws from collector
        collector = KinoDataCollector()
        draws = collector.fetch_latest_draws(num_draws=1)
        
        if not draws:
            return False, "No draws available for analysis"
            
        # Use handler to perform analysis
        handler = DrawHandler()
        from src.draw_handler import perform_complete_analysis
        
        analysis_success = perform_complete_analysis(draws)
        analysis_duration = (datetime.now(TIMEZONE) - start_time).total_seconds()
        
        if analysis_success:
            print(f"[Automation] Analysis completed successfully in {analysis_duration:.2f} seconds.")
            return True, {
                "duration": analysis_duration,
                "draws_analyzed": len(draws)
            }
        else:
            return False, "Failed to complete analysis"
        
    except Exception as e:
        error_msg = f"Error in analysis: {str(e)}"
        print(f"[Automation] {error_msg}")
        return False, error_msg

def generate_prediction_operation(for_draw_time=None):
    """
    Generate ML prediction for next draw.
    
    This operation corresponds to the PREDICTING state in the state machine.
    
    Args:
        for_draw_time: The draw time to predict for
        
    Returns:
        tuple: (success, message_or_prediction_data)
    """
    try:
        if for_draw_time:
            print(f"\n[Automation] Generating ML prediction for draw at {for_draw_time.strftime('%H:%M:%S')}")
        else:
            print("\n[Automation] Generating ML prediction for next draw...")
        
        start_time = datetime.now(TIMEZONE)
        
        # Import handler after path setup
        from src.draw_handler import DrawHandler
        
        # Initialize handler
        handler = DrawHandler()
        
        # Use handler to generate prediction
        predictions, probabilities, analysis = handler.handle_prediction_pipeline()
        prediction_duration = (datetime.now(TIMEZONE) - start_time).total_seconds()
        
        if predictions is not None:
            print(f"[Automation] Generated prediction: {sorted(predictions)} in {prediction_duration:.2f} seconds")
            
            # Format draw time for logging
            if for_draw_time:
                # Format time in UTC+2
                time_str = for_draw_time.astimezone(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
                print(f"[Automation] For draw at: {time_str} (UTC+2)")
                
            return True, {
                "predictions": predictions, 
                "probabilities": probabilities, 
                "analysis": analysis,
                "duration": prediction_duration,
                "target_draw_time": for_draw_time
            }
        else:
            return False, "Failed to generate prediction"
        
    except Exception as e:
        error_msg = f"Error generating prediction: {str(e)}"
        print(f"[Automation] {error_msg}")
        return False, error_msg

def evaluate_prediction_operation():
    """
    Evaluate past predictions against actual draws.
    
    This operation corresponds to the EVALUATING state in the state machine.
    
    Returns:
        tuple: (success, message_or_stats)
    """
    try:
        print("\n[Automation] Evaluating past predictions...")
        start_time = datetime.now(TIMEZONE)
        
        # Import evaluator after path setup
        from src.prediction_evaluator import PredictionEvaluator
        
        # Create evaluator and run evaluation
        evaluator = PredictionEvaluator()
        evaluator.evaluate_past_predictions()
        
        # Get performance stats
        stats = evaluator.get_performance_stats()
        evaluation_duration = (datetime.now(TIMEZONE) - start_time).total_seconds()
        
        if stats:
            print(f"\n[Automation] Evaluation completed in {evaluation_duration:.2f} seconds")
            print("\n[Automation] Evaluation Summary:")
            print(f"- Total predictions evaluated: {stats.get('total_predictions', 0)}")
            print(f"- Average correct numbers: {stats.get('average_correct', 0):.1f}")
            print(f"- Best prediction: {stats.get('best_prediction', 0)} correct numbers")
            print(f"- Average accuracy: {stats.get('average_accuracy', 0):.1f}%")
            if 'recent_trend' in stats:
                trend = stats['recent_trend']
                print(f"- Recent trend: {'Improving' if trend > 0 else 'Declining'} ({trend:.3f})")
            
            # Add duration to stats
            stats['duration'] = evaluation_duration
            
            return True, stats
        else:
            return False, "No evaluation statistics available"
        
    except Exception as e:
        error_msg = f"Error in evaluation: {str(e)}"
        print(f"[Automation] {error_msg}")
        return False, error_msg

def run_continuous_learning():
    """
    Run the continuous learning cycle to improve predictions.
    
    This operation corresponds to the LEARNING state in the state machine.
    
    Returns:
        tuple: (success, message_or_metrics)
    """
    try:
        print("\n[Automation] Running continuous learning cycle...")
        start_time = datetime.now(TIMEZONE)
        
        # Import handler after path setup
        from src.draw_handler import DrawHandler
        
        # Initialize handler
        handler = DrawHandler()
        
        # Run continuous learning cycle
        success = handler.run_continuous_learning_cycle()
        learning_duration = (datetime.now(TIMEZONE) - start_time).total_seconds()
        
        if success:
            # Get metrics to display
            metrics = handler.get_learning_metrics()
            metrics['duration'] = learning_duration
            
            print(f"\n[Automation] Continuous Learning completed in {learning_duration:.2f} seconds")
            print("\n[Automation] Continuous Learning Results:")
            print(f"- Learning cycles completed: {metrics.get('cycles_completed', 0)}")
            
            if metrics.get('current_accuracy') is not None:
                print(f"- Current prediction accuracy: {metrics['current_accuracy']:.2f}%")
                
            if metrics.get('improvement_rate') is not None:
                print(f"- Total improvement: {metrics['improvement_rate']:.2f}%")
                
            print("\n[Automation] Most recent adjustments:")
            if metrics.get('last_adjustments') and len(metrics['last_adjustments']) > 0:
                for adj in metrics['last_adjustments']:
                    print(f"- {adj}")
            else:
                print("- No adjustments made")
                
            return True, metrics
        else:
            return False, {
                "error": "Continuous learning cycle failed or made no improvements",
                "duration": learning_duration
            }
        
    except Exception as e:
        error_msg = f"Error in continuous learning: {str(e)}"
        print(f"[Automation] {error_msg}")
        return False, error_msg

if __name__ == "__main__":
    # For testing the operations module
    print("Testing operations module...")
    test_operation()
    print("Running data collection...")
    collect_data_operation(num_draws=2)
    print("Running analysis...")
    analyze_data_operation()
    print("Generating prediction...")
    generate_prediction_operation()
    print("Evaluating predictions...")
    evaluate_prediction_operation()
    print("Running continuous learning...")
    run_continuous_learning()
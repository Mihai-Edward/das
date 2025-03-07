# File: automation/cycle_manager.py
import os
import sys
import time
import logging
from datetime import datetime, timedelta

import pytz

# Add src directory directly to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)  # Add src directory directly
sys.path.insert(0, project_root)  # Also add project root

# Import required modules from main project (these will now be found in src directory)
from config.paths import PATHS, ensure_directories
from draw_handler import DrawHandler, save_draw_to_csv, perform_complete_analysis, train_and_predict
from data_collector_selenium import KinoDataCollector
from prediction_evaluator import PredictionEvaluator

# Import components from the automation package
from automation.scheduler import DrawScheduler, get_next_draw_time, get_seconds_until, get_formatted_time_remaining
from automation.operations import fetch_latest_draws, perform_analysis, generate_prediction, evaluate_prediction

class PredictionCycleManager:
    """
    Manages the automated prediction cycle that follows lottery draws.
    Orchestrates data fetching, analysis, prediction, and evaluation with retry logic.
    """
    
    def __init__(self):
        """Initialize the cycle manager with default settings."""
        # Core configuration
        self.consecutive_failures = 0
        self.max_failures = 5
        self.retries_per_operation = 3
        self.retry_delay = 30  # seconds
        
        # State management
        self.current_state = 'initializing'
        self.cycle_count = 0
        self.last_successful_cycle = None
        
        # Initialize scheduler
        self.scheduler = DrawScheduler(draw_interval_minutes=5, post_draw_wait_seconds=50)
        
        # Create status file
        self.status_file = os.path.join(current_dir, 'automation_status.txt')
        self._update_status("Initialized")

    def _update_status(self, status):
        """Update automation status file"""
        try:
            with open(self.status_file, 'w') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp}: {status}\n")
        except Exception as e:
            print(f"Error updating status: {e}")
    
    def _run_with_retry(self, operation_func, *args, **kwargs):
        """Run an operation with retry logic."""
        operation_name = operation_func.__name__
        
        for attempt in range(1, self.retries_per_operation + 1):
            self._update_status(f"Running {operation_name} (attempt {attempt}/{self.retries_per_operation})")
            
            try:
                success, result = operation_func(*args, **kwargs)
                
                if success:
                    self._update_status(f"{operation_name} completed successfully")
                    return True, result
                
                error_msg = result if isinstance(result, str) else "Operation returned failure"
                self._update_status(f"{operation_name} failed: {error_msg}")
                
            except Exception as e:
                self._update_status(f"{operation_name} raised exception: {str(e)}")
                success = False
            
            # If we get here, the operation failed
            if attempt < self.retries_per_operation:
                retry_msg = f"Retrying {operation_name} in {self.retry_delay} seconds..."
                self._update_status(retry_msg)
                time.sleep(self.retry_delay)
            else:
                self._update_status(f"All {self.retries_per_operation} attempts at {operation_name} failed")
        
        # All attempts failed
        return False, None

    def evaluate_results(self):
        """Evaluate prediction accuracy"""
        try:
            self.logger.info("[Evaluation] Evaluating predictions...")
            self.evaluator.evaluate_past_predictions()
            self.logger.info("[Evaluation] Evaluation complete")
            return True
        except Exception as e:
            self.logger.error(f"[Evaluation] Error during evaluation: {e}")
            return False
    def collect_data(self):
        """Collect latest draw data with retries"""
        for attempt in range(self.fetch_retries):
            try:
                self.logger.info("[Data Collection] Fetching latest draws...")
                draws = self.collector.fetch_latest_draws()
                if draws:
                    self.collector.sort_historical_draws()
                    self.last_collection_time = datetime.now(pytz.UTC)
                    self.logger.info(f"[Data Collection] Successfully collected {len(draws)} draws")
                    return True
            except Exception as e:
                self.logger.error(f"[Data Collection] Attempt {attempt + 1} failed: {e}")
                if attempt < self.fetch_retries - 1:
                    time.sleep(self.retry_delay)
        return False
    def generate_prediction(self):
        """Generate prediction for next draw"""
        try:
            self.logger.info("[Prediction] Generating ML prediction...")
            predictions, probabilities, analysis = self.handler.handle_prediction_pipeline()
            
            if predictions is not None:
                self.last_prediction_time = datetime.now(pytz.UTC)
                self.logger.info(f"[Prediction] Generated numbers: {sorted(predictions)}")
                return predictions, probabilities, analysis
            return None, None, None
        except Exception as e:
            self.logger.error(f"[Prediction] Error generating prediction: {e}")
            return None, None, None    
def run_cycle(self):
    """Run the complete automation cycle"""
    while True:
            try:
                current_time = datetime.now(pytz.UTC)
                next_draw = self.scheduler.get_next_draw_time(current_time)
                
                # 1. Collect Data
                if not self.collect_data():
                    self.failures += 1
                    if self.failures >= self.max_failures:
                        raise Exception("Maximum failures reached in data collection")
                    continue
                
                # 2. Generate Prediction
                predictions, probabilities, analysis = self.generate_prediction()
                if predictions is None:
                    self.failures += 1
                    if self.failures >= self.max_failures:
                        raise Exception("Maximum failures reached in prediction")
                    continue
                
                # 3. Wait for draw time
                wait_time = (next_draw - current_time).total_seconds()
                if wait_time > 0:
                    self.logger.info(f"Waiting {wait_time:.0f} seconds until next draw...")
                    time.sleep(wait_time)
                
                # 4. Wait post-draw interval
                self.logger.info(f"Waiting {self.scheduler.post_draw_wait_seconds} seconds for draw results...")
                time.sleep(self.scheduler.post_draw_wait_seconds)
                
                # 5. Evaluate Results
                self.evaluate_results()
                
                # Reset failure count on successful cycle
                self.failures = 0
                self.cycle_count += 1
                
                self.logger.info(f"Completed cycle {self.cycle_count}")
                
            except Exception as e:
                self.logger.error(f"Error in automation cycle: {e}")
                self.failures += 1
                if self.failures >= self.max_failures:
                    raise Exception(f"Maximum failures ({self.max_failures}) reached")
                time.sleep(self.retry_delay)
    
    
    def get_status(self):
        """Get current automation status"""
        return {
            'cycles_completed': self.cycle_count,
            'current_failures': self.failures,
            'last_collection': self.last_collection_time,
            'last_prediction': self.last_prediction_time,
            'next_draw': self.scheduler.get_next_draw_time(),
            'max_failures': self.max_failures,
            'retry_delay': self.retry_delay
        }

if __name__ == "__main__":
    # Simple test of the cycle manager
    print("Testing cycle manager...")
    manager = PredictionCycleManager()
    
    # Print scheduler details for next draw
    next_draw = manager.scheduler.get_next_draw_time()
    eval_time = manager.scheduler.get_evaluation_time(next_draw)
    
    print(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Next draw at: {next_draw.strftime('%H:%M:%S')}")
    print(f"Evaluate at: {eval_time.strftime('%H:%M:%S')}")
    
    # Don't actually run the cycle in test mode
    print("Test complete. Import this module and call run_cycle() to start automation.")
# File: automation/cycle_manager.py
import os
import sys
import time
import logging
from datetime import datetime, timedelta

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
    
    def run_cycle(self):
        """Run a complete prediction cycle with error handling."""
        try:
            self._update_status("Starting automated prediction cycles")
            
            while self.consecutive_failures < self.max_failures:
                try:
                    # Record start time and increment cycle counter
                    cycle_start = datetime.now()
                    self.cycle_count += 1
                    
                    self._update_status(f"Starting cycle #{self.cycle_count}")
                    
                    # Step 1: Fetch latest draws (Option 3)
                    self.current_state = 'fetching'
                    fetch_success, draws = self._run_with_retry(fetch_latest_draws)
                    if not fetch_success:
                        self.consecutive_failures += 1
                        self._update_status(f"Fetch failed. Consecutive failures: {self.consecutive_failures}")
                        continue
                    
                    # Step 2: Perform analysis (Option 8)
                    self.current_state = 'analyzing'
                    analyze_success, _ = self._run_with_retry(perform_analysis, draws)
                    if not analyze_success:
                        self.consecutive_failures += 1
                        self._update_status(f"Analysis failed. Consecutive failures: {self.consecutive_failures}")
                        continue
                    
                    # Step 3: Generate prediction (Option 9)
                    self.current_state = 'predicting'
                    predict_success, prediction_data = self._run_with_retry(generate_prediction)
                    if not predict_success:
                        self.consecutive_failures += 1
                        self._update_status(f"Prediction failed. Consecutive failures: {self.consecutive_failures}")
                        continue
                        
                    # Calculate time until next draw
                    next_draw_time = self.scheduler.get_next_draw_time(cycle_start)
                    one_minute_before = next_draw_time - timedelta(minutes=1)
                    
                    # Wait until 1 minute before next draw if needed
                    wait_seconds = self.scheduler.seconds_until_time(one_minute_before)
                    if wait_seconds > 0:
                        self.current_state = 'waiting_for_draw'
                        self._update_status(f"Waiting until 1 minute before next draw at {next_draw_time.strftime('%H:%M:%S')}")
                        time.sleep(wait_seconds)
                        
                    # Wait for draw to complete
                    self.current_state = 'waiting_post_draw'
                    evaluation_time = self.scheduler.get_evaluation_time(next_draw_time)
                    wait_seconds = self.scheduler.seconds_until_time(evaluation_time)
                    
                    if wait_seconds > 0:
                        self._update_status(f"Waiting {wait_seconds} seconds for draw completion")
                        time.sleep(wait_seconds)
                    
                    # Step 5: Evaluate prediction (Option 10)
                    self.current_state = 'evaluating'
                    eval_success, eval_data = self._run_with_retry(evaluate_prediction)
                    if not eval_success:
                        self.consecutive_failures += 1
                        self._update_status(f"Evaluation failed. Consecutive failures: {self.consecutive_failures}")
                        continue
                    
                    # Successful cycle completion
                    self.consecutive_failures = 0  # Reset failure counter on success
                    self.last_successful_cycle = datetime.now()
                    self._update_status(f"Cycle {self.cycle_count} completed successfully")
                    
                except Exception as e:
                    self.consecutive_failures += 1
                    self._update_status(f"Error in cycle {self.cycle_count}: {str(e)}")
                    self._update_status(f"Consecutive failures: {self.consecutive_failures}")
                
                # Check if we've hit the failure limit
                if self.consecutive_failures >= self.max_failures:
                    self._update_status(f"Stopping after {self.consecutive_failures} consecutive failures")
                    break
        
        except KeyboardInterrupt:
            self._update_status("Automation stopped by user (Keyboard Interrupt)")
        except Exception as e:
            self._update_status(f"Critical error in cycle manager: {str(e)}")
        finally:
            self._update_status("Cycle manager shutting down")
    
    def get_status(self):
        """Get the current status of the cycle manager."""
        return {
            'state': self.current_state,
            'cycle_count': self.cycle_count,
            'consecutive_failures': self.consecutive_failures,
            'last_successful_cycle': self.last_successful_cycle,
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
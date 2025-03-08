# automation/cycle_manager.py

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
from automation.scheduler import DrawScheduler
from config.paths import PATHS, ensure_directories

# Define the standard timezone to use
TIMEZONE = pytz.timezone('Europe/Bucharest')  # UTC+2

class PredictionCycleManager:
    def __init__(self):
        """Initialize the cycle manager with default settings."""
        # Status tracking
        self.active = True
        self.status = "initializing"
        self.last_success = None
        self.last_error = None
        self.consecutive_failures = 0
        
        # Configuration
        self.max_failures = 5
        self.retry_delay = 30  # seconds
        self.fetch_retries = 3
        self.continuous_learning_enabled = False  # Default to disabled
        
        # Initialize scheduler
        self.scheduler = DrawScheduler()
        
        # Log initialization
        self._log_status("Prediction cycle manager initialized")
        
    def get_status(self):
        """Get the current status of the cycle manager."""
        return {
            'active': self.active,
            'status': self.status,
            'last_success': self.last_success,
            'last_error': self.last_error,
            'consecutive_failures': self.consecutive_failures,
            'next_draw': self.scheduler.get_next_draw_time(),
            'next_eval': self.scheduler.get_evaluation_time(self.scheduler.get_next_draw_time())
        }
        
    def _log_status(self, message):
        """Log a status message with timestamp."""
        # Use UTC+2 timezone for logs
        timestamp = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")
        
    def _handle_failure(self, stage, message):
        """Handle a failure during operation."""
        self.consecutive_failures += 1
        self.last_error = {'stage': stage, 'message': message, 'time': datetime.now(TIMEZONE)}
        self._log_status(f"Failure in {stage}: {message}")
        self._log_status(f"Consecutive failures: {self.consecutive_failures}/{self.max_failures}")
        
        # Exit if max failures reached
        if self.consecutive_failures >= self.max_failures:
            self._log_status("Maximum consecutive failures reached, stopping cycle")
            self.active = False
        
        # Add delay before retrying
        time.sleep(self.retry_delay)
        
    def collect_data(self):
        """Collect data for prediction."""
        from automation.operations import collect_data_operation
        success, message = collect_data_operation(num_draws=self.fetch_retries * 3)
        return success, message
        
    def analyze_data(self):
        """Analyze collected data."""
        from automation.operations import analyze_data_operation
        success, message = analyze_data_operation()
        return success, message
        
    def generate_prediction(self, for_draw_time=None):
        """Generate prediction for next draw."""
        from automation.operations import generate_prediction_operation
        success, message = generate_prediction_operation(for_draw_time=for_draw_time)
        return success, message
        
    def evaluate_prediction(self):
        """Evaluate past predictions."""
        from automation.operations import evaluate_prediction_operation
        success, message = evaluate_prediction_operation()
        
        # Track status and return results
        if success:
            self.last_success = datetime.now(TIMEZONE)
            self.consecutive_failures = 0
        return success, message
        
    def run_continuous_learning(self):
        """Run the continuous learning cycle after evaluation."""
        from automation.operations import run_continuous_learning
        success, message = run_continuous_learning()
        
        # Track status and return results
        if success:
            self.last_success = datetime.now(TIMEZONE)
            self.consecutive_failures = 0
            return True, "Continuous learning completed successfully"
        else:
            # Don't increment failure counter for learning failures
            # as this is not a critical operation
            return False, f"Continuous learning failed: {message}"
        
    def run_cycle(self):
        """Run the main prediction cycle."""
        # Start with data collection immediately
        self.status = "predicting"
        self._log_status("Starting initial prediction cycle")
        
        # Immediate data collection at startup
        self._log_status("Collecting initial data...")
        success, message = self.collect_data()
        if not success:
            self._handle_failure("initial_data_collection", message)
        else:
            # Continue with analysis
            self._log_status("Analyzing initial data...")
            success, message = self.analyze_data()
            if not success:
                self._handle_failure("initial_analysis", message)
            else:
                # Generate prediction
                self._log_status("Generating initial prediction...")
                next_draw_time = self.scheduler.get_next_draw_time()
                success, message = self.generate_prediction(for_draw_time=next_draw_time)
                if not success:
                    self._handle_failure("initial_prediction", message)
                else:
                    # Run continuous learning if enabled
                    if self.continuous_learning_enabled:
                        self._log_status("Running initial continuous learning...")
                        success, message = self.run_continuous_learning()
                        if success:
                            self._log_status("Initial continuous learning completed successfully")
                        else:
                            self._log_status(f"Note: Initial continuous learning skipped or failed: {message}")
                    
                    # Success! Record and log
                    self.last_success = datetime.now(TIMEZONE)
                    self.consecutive_failures = 0
        
        # Main cycle loop
        while self.active and self.consecutive_failures < self.max_failures:
            try:
                # Get the next draw time
                next_draw_time = self.scheduler.get_next_draw_time()
                evaluation_time = self.scheduler.get_evaluation_time(next_draw_time)
                
                # Calculate wait time
                now = datetime.now(TIMEZONE)
                
                # Debug logging
                self._log_status(f"Debug: now = {now.strftime('%H:%M:%S')}, next_draw = {next_draw_time.strftime('%H:%M:%S')}, eval = {evaluation_time.strftime('%H:%M:%S')}")
                
                # 1. Check if we should evaluate after a draw just happened (with 2-second tolerance)
                if now >= evaluation_time - timedelta(seconds=2) and now < next_draw_time + timedelta(minutes=2):
                    self.status = "evaluating"
                    self._log_status(f"Evaluating draw at {next_draw_time.strftime('%H:%M:%S')}")
                    
                    # Run evaluation
                    success, message = self.evaluate_prediction()
                    if not success:
                        self._handle_failure("evaluation", message)
                    else:
                        self._log_status("Evaluation complete")
                    
                    # Get the next draw time after evaluation
                    next_draw_time = self.scheduler.get_next_draw_time()
                    
                    # After evaluation, immediately collect data for the next prediction
                    self.status = "predicting"
                    self._log_status("Starting prediction cycle for next draw")
                    
                    # A) Collect latest draw data
                    self._log_status("Collecting data...")
                    success, message = self.collect_data()
                    if not success:
                        self._handle_failure("data_collection", message)
                        continue
                        
                    # B) Analyze collected data
                    self._log_status("Analyzing data...")
                    success, message = self.analyze_data()
                    if not success:
                        self._handle_failure("analysis", message)
                        continue
                
                    # C) Generate prediction
                    self._log_status("Generating prediction...")
                    next_next_draw = self.scheduler.get_next_draw_time(next_draw_time)
                    success, message = self.generate_prediction(for_draw_time=next_next_draw)
                    if not success:
                        self._handle_failure("prediction", message)
                        continue
                        
                    # D) Run continuous learning if enabled
                    if self.continuous_learning_enabled:
                        self._log_status("Running continuous learning...")
                        success, message = self.run_continuous_learning()
                        if success:
                            self._log_status("Continuous learning completed successfully")
                        else:
                            self._log_status(f"Note: Continuous learning skipped or failed: {message}")
                    
                    # Success! Record and log
                    self.last_success = now
                    self.consecutive_failures = 0
                    
                    # Wait until just after the next draw time
                    self._log_status(f"Prediction complete, waiting for next draw at {next_next_draw.strftime('%H:%M:%S')}")
                                        
                # 3. Otherwise, just wait a bit
                else:
                    self.status = "waiting"
                    time_to_next_draw = (next_draw_time - now).total_seconds()
                    
                    if time_to_next_draw > 60:  # If more than a minute to wait
                        self._log_status(f"Waiting {int(time_to_next_draw/60)} minutes {int(time_to_next_draw%60)} seconds until next draw")
                    else:
                        self._log_status(f"Waiting {int(time_to_next_draw)} seconds until next draw")
                
                # Sleep for a bit before checking again  
                time.sleep(10)  # Check every 10 seconds instead of 30
                
            except Exception as e:
                self._handle_failure("cycle", str(e))
                time.sleep(self.retry_delay)
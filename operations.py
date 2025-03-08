
import os
import sys
import time
from datetime import datetime, timedelta
import pytz
from enum import Enum

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

# Define states as enum for clarity and type safety
class PredictionState(Enum):
    FETCHING = "fetching"
    ANALYZING = "analyzing"
    PREDICTING = "predicting"
    LEARNING = "learning"
    WAITING = "waiting"
    PREPARING_TO_EVALUATE = "preparing_to_evaluate"
    EVALUATING = "evaluating"

class PredictionCycleManager:
    def __init__(self):
        """Initialize the cycle manager with default settings."""
        # Status tracking
        self.active = True
        self.state = PredictionState.FETCHING  # Start with fetching state
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
        
        # Initialize prediction data
        self.current_prediction = None
        self.current_probabilities = None
        
        # Collected draws for analysis
        self.collected_draws = None
        
        # Log initialization
        self._log_status("Prediction cycle manager initialized")
        
    def get_status(self):
        """Get the current status of the cycle manager."""
        return {
            'active': self.active,
            'state': self.state.value,
            'last_success': self.last_success,
            'last_error': self.last_error,
            'consecutive_failures': self.consecutive_failures,
            'next_draw': self.scheduler.get_next_draw_time(),
            'next_eval': self.scheduler.get_evaluation_time(self.scheduler.get_next_draw_time())
        }
        
    def _log_status(self, message):
        """Log a status message with timestamp and current state."""
        # Use UTC+2 timezone for logs
        timestamp = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
        state_info = f"[{self.state.value}]" if hasattr(self, 'state') else ""
        print(f"[{timestamp}] {state_info} {message}")
        
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
        if success:
            self.collected_draws = message["draws"]  # Store collected draws for analysis
        return success, message
        
    def analyze_data(self):
        """Analyze collected data."""
        from automation.operations import analyze_data_operation
        success, message = analyze_data_operation(self.collected_draws)  # Pass collected draws to the function
        return success, message
        
    def generate_prediction(self, for_draw_time=None):
        """Generate prediction for next draw."""
        from automation.operations import generate_prediction_operation
        success, result = generate_prediction_operation(for_draw_time=for_draw_time)
        
        # Store prediction and probabilities if successful
        if success and isinstance(result, dict):
            self.current_prediction = result.get("predictions")
            self.current_probabilities = result.get("probabilities")
            
        return success, result
        
    def evaluate_prediction(self):
        """Evaluate past predictions."""
        from automation.operations import evaluate_prediction_operation
        success, message = evaluate_prediction_operation()
        
        # Track status
        if success:
            self.last_success = datetime.now(TIMEZONE)
            self.consecutive_failures = 0
            
        return success, message
        
    def run_continuous_learning(self):
        """Run the continuous learning cycle after evaluation."""
        from automation.operations import run_continuous_learning
        success, message = run_continuous_learning()
        
        # Track status
        if success:
            self.last_success = datetime.now(TIMEZONE)
            
        # Don't increment failure counter for learning failures
        # as this is not a critical operation
        return success, message
        
    def run_cycle(self):
        """Run the main prediction cycle using state machine approach."""
        self._log_status("Starting prediction cycle with state machine approach")
        
        # Main cycle loop
        while self.active and self.consecutive_failures < self.max_failures:
            try:
                # Log current state
                self._log_status(f"Current state: {self.state.value}")
                
                # Handle current state
                if self.state == PredictionState.FETCHING:
                    self._handle_fetching_state()
                elif self.state == PredictionState.ANALYZING:
                    self._handle_analyzing_state()
                elif self.state == PredictionState.PREDICTING:
                    self._handle_predicting_state()
                elif self.state == PredictionState.LEARNING:
                    self._handle_learning_state()
                elif self.state == PredictionState.WAITING:
                    self._handle_waiting_state()
                elif self.state == PredictionState.PREPARING_TO_EVALUATE:
                    self._handle_preparing_to_evaluate_state()
                elif self.state == PredictionState.EVALUATING:
                    self._handle_evaluating_state()
                
                # Brief pause between state checks
                time.sleep(3)
                
            except Exception as e:
                self._handle_failure("cycle", str(e))
                time.sleep(self.retry_delay)

    def _handle_fetching_state(self):
        """Handle the fetching state operations and transitions."""
        self._log_status("Collecting data...")
        success, message = self.collect_data()
        
        if success:
            # On success, transition to ANALYZING state
            self.state = PredictionState.ANALYZING
        else:
            # On failure, retry in current state
            self._handle_failure("data_collection", message)
    
    def _handle_analyzing_state(self):
        """Handle the analyzing state operations and transitions."""
        self._log_status("Analyzing data...")
        success, message = self.analyze_data()
        
        if success:
            # On success, transition to PREDICTING state
            self.state = PredictionState.PREDICTING
        else:
            # On failure, retry in current state
            self._handle_failure("analysis", message)
    
    def _handle_predicting_state(self):
        """Handle the predicting state operations and transitions."""
        self._log_status("Generating prediction...")
        next_draw_time = self.scheduler.get_next_draw_time()
        success, result = self.generate_prediction(for_draw_time=next_draw_time)
        
        if success:
            # On success, transition to LEARNING or WAITING state based on configuration
            if self.continuous_learning_enabled:
                self.state = PredictionState.LEARNING
            else:
                self.state = PredictionState.WAITING
                self._log_status(f"Prediction complete. Waiting for next draw at {next_draw_time.strftime('%H:%M:%S')}")
        else:
            # On failure, retry in current state
            self._handle_failure("prediction", str(result))
    
    def _handle_learning_state(self):
        """Handle the continuous learning state operations and transitions."""
        if self.continuous_learning_enabled:
            self._log_status("Running continuous learning...")
            success, message = self.run_continuous_learning()
            
            if success:
                self._log_status("Continuous learning completed successfully")
            else:
                self._log_status(f"Note: Continuous learning failed: {message}")
                # Learning failures don't stop the cycle
        else:
            self._log_status("Continuous learning disabled, skipping")
            
        # Always transition to WAITING state after learning (success or failure)
        self.state = PredictionState.WAITING
        next_draw_time = self.scheduler.get_next_draw_time()
        self._log_status(f"Waiting for next draw at {next_draw_time.strftime('%H:%M:%S')}")
    
    def _handle_waiting_state(self):
        """Handle the waiting state operations and transitions."""
        next_draw_time = self.scheduler.get_next_draw_time()
        evaluation_time = self.scheduler.get_evaluation_time(next_draw_time)
        now = datetime.now(TIMEZONE)
        
        # Calculate time until evaluation
        time_to_evaluate = (evaluation_time - now).total_seconds()
        
        # Transition to PREPARING_TO_EVALUATE if less than 10 seconds to draw
        if time_to_evaluate <= 10:
            self._log_status("Preparing to evaluate draw")
            self.state = PredictionState.PREPARING_TO_EVALUATE
            return
            
        # Otherwise, display waiting status
        time_to_next_draw = (next_draw_time - now).total_seconds()
        if time_to_next_draw > 60:  # If more than a minute
            self._log_status(f"Waiting {int(time_to_next_draw/60)} minutes {int(time_to_next_draw%60)} seconds until next draw")
        else:
            self._log_status(f"Waiting {int(time_to_next_draw)} seconds until next draw")
        
        # Sleep longer during waiting to reduce log spam
        time.sleep(11)
    
    def _handle_preparing_to_evaluate_state(self):
        """Handle the preparing to evaluate state operations and transitions."""
        self._log_status("Preparing for evaluation")
        time.sleep(11)  # Wait for 10 seconds to ensure the draw has occurred
        self.state = PredictionState.EVALUATING
    
    def _handle_evaluating_state(self):
        """Handle the evaluating state operations and transitions."""
        self._log_status("Evaluating predictions...")
        success, message = self.evaluate_prediction()
        
        if success:
            self._log_status("Evaluation complete")
            self.last_success = datetime.now(TIMEZONE)
            self.consecutive_failures = 0
        else:
            self._handle_failure("evaluation", str(message))
            # Evaluation failures don't stop the cycle
        
        # Always transition back to FETCHING to start the cycle again
        self.state = PredictionState.FETCHING
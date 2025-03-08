import os
import sys
import signal
import traceback
from datetime import datetime, timedelta
import pytz
import argparse

# Add the project root and src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now try the imports
try:
    from automation.cycle_manager import PredictionCycleManager, PredictionState
    from automation.scheduler import DrawScheduler
    from automation.operations import test_operation
    from config.paths import ensure_directories
    from src.data_collector_selenium import KinoDataCollector
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    sys.exit(1)

def ensure_models_trained():
    """Ensure models are properly trained before proceeding."""
    print("\nChecking model state...")
    
    from src.lottery_predictor import LotteryPredictor
    from src.draw_handler import DrawHandler
    
    handler = DrawHandler()
    try:
        # Check if model exists and passes validation
        predictor = LotteryPredictor()
        model_path = handler._get_latest_model()
        
        if model_path and predictor.load_models(model_path):
            is_valid, message = predictor.validate_model_state()
            if is_valid:
                print("✓ Models validated successfully")
                return True
        
        # Model doesn't exist or validation failed, train new models
        print("! Models need training. Training new models...")
        success = handler.train_ml_models(force_retrain=True)
        if success:
            print("✓ Models trained successfully")
            return True
        else:
            print("✗ Model training failed")
            return False
    
    except Exception as e:
        print(f"✗ Error checking/training models: {e}")
        import traceback
        traceback.print_exc()
        return False

def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Automated lottery prediction system runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic configuration arguments
    parser.add_argument(
        '--max-failures',
        type=int,
        default=5,
        help='Maximum consecutive failures before stopping'
    )
    
    parser.add_argument(
        '--retry-delay',
        type=int,
        default=30,
        help='Delay in seconds between retry attempts'
    )
    
    parser.add_argument(
        '--post-draw-wait',
        type=int,
        default=50,
        help='Time to wait in seconds after draw before evaluating'
    )
    
    parser.add_argument(
        '--fetch-retries',
        type=int,
        default=3,
        help='Number of retries for fetching draw results'
    )
    
    # Special mode arguments
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode without starting the cycle'
    )
    
    parser.add_argument(
        '--debug-models',
        action='store_true',
        help='Run model debugging mode'
    )

    # New arguments for state machine control
    parser.add_argument(
        '--enable-learning',
        action='store_true',
        help='Enable continuous learning to improve model based on past predictions'
    )
    
    parser.add_argument(
        '--initial-state',
        choices=['fetching', 'analyzing', 'predicting', 'learning', 'waiting', 'evaluating'],
        default='fetching',
        help='Set the initial state of the prediction cycle (for testing purposes)'
    )
    
    return parser

def get_formatted_time_remaining(target_time):
    """Calculate and format time remaining until target time."""
    # Use Europe/Bucharest timezone (UTC+2)
    now = datetime.now(pytz.timezone('Europe/Bucharest'))
    if target_time <= now:
        return "0m 0s"
    
    delta = target_time - now
    minutes = int(delta.total_seconds() // 60)
    seconds = int(delta.total_seconds() % 60)
    return f"{minutes}m {seconds}s"

def display_header():
    """Display welcome message and system information."""
    print("\n" + "="*50)
    print("      AUTOMATED LOTTERY PREDICTION SYSTEM")
    print("="*50)
    
    # Use Europe/Bucharest timezone (UTC+2)
    now = datetime.now(pytz.timezone('Europe/Bucharest'))
    scheduler = DrawScheduler()
    next_draw = scheduler.get_next_draw_time(now)
    eval_time = scheduler.get_evaluation_time(next_draw)
    
    print(f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M:%S')} (UTC+2)")
    print(f"Next draw at: {next_draw.strftime('%H:%M:%S')}")
    print(f"Next evaluation at: {eval_time.strftime('%H:%M:%S')}")
    print(f"Time until next draw: {get_formatted_time_remaining(next_draw)}")
    print(f"System directory: {os.path.dirname(__file__)}")
    
    # Check if required directories exist
    try:
        ensure_directories()
        print("Directory structure verified.")
    except Exception as e:
        print(f"Warning: Issue with directories - {str(e)}")
    
    print("\nPress Ctrl+C at any time to stop the automation.")
    print("="*50 + "\n")

def graceful_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\n\nReceived shutdown signal. Cleaning up...")
    print("Please wait for current operation to complete...")
    sys.exit(0)

def debug_models():
    """Debug model loading and validation issues."""
    try:
        print("\n===== MODEL DEBUG MODE =====")
        
        from src.lottery_predictor import LotteryPredictor
        from src.draw_handler import DrawHandler
        
        # Check model files
        print("\nChecking model files:")
        handler = DrawHandler()
        model_path = handler._get_latest_model()
        print(f"Latest model path: {model_path}")
        
        if model_path:
            model_files = [
                f"{model_path}_prob_model.pkl",
                f"{model_path}_pattern_model.pkl",
                f"{model_path}_scaler.pkl"
            ]
            
            for file in model_files:
                if os.path.exists(file):
                    file_size = os.path.getsize(file)
                    print(f"✓ {os.path.basename(file)} - {file_size} bytes")
                else:
                    print(f"✗ {os.path.basename(file)} - Missing")
        
        # Try manual model loading
        print("\nTrying direct model loading:")
        predictor = LotteryPredictor()
        loaded = predictor.load_models(model_path)
        print(f"Model loading result: {loaded}")
        
        # Validate model state
        print("\nValidating model state:")
        is_valid, message = predictor.validate_model_state()
        print(f"Model valid: {is_valid}")
        print(f"Validation message: {message}")
        
        # Check model attributes
        print("\nModel attributes:")
        if hasattr(predictor, 'probabilistic_model'):
            print(f"Probabilistic model type: {type(predictor.probabilistic_model).__name__}")
            if hasattr(predictor.probabilistic_model, 'class_prior_'):
                print("✓ class_prior_ attribute exists")
            else:
                print("✗ class_prior_ attribute missing")
        
        if hasattr(predictor, 'pattern_model'):
            print(f"Pattern model type: {type(predictor.pattern_model).__name__}")
            if hasattr(predictor.pattern_model, 'coefs_'):
                print("✓ coefs_ attribute exists")
            else:
                print("✗ coefs_ attribute missing")
        
        print("\nAttempting to train models:")
        try:
            # Load some data for training
            import pandas as pd
            data_file = os.path.join(src_dir, 'historical_draws.csv')
            historical_data = pd.read_csv(data_file)
            print(f"Loaded {len(historical_data)} records for training")
            
            # Try training
            features, labels = predictor.prepare_data(historical_data)
            if features is not None and labels is not None:
                print(f"Prepared features shape: {features.shape}")
                print(f"Prepared labels shape: {labels.shape}")
                
                success = predictor.train_models(features, labels)
                print(f"Model training result: {success}")
                
                # Try validating again
                is_valid, message = predictor.validate_model_state()
                print(f"After training - Model valid: {is_valid}")
                print(f"Validation message: {message}")
        except Exception as e:
            print(f"Error during training: {e}")
        
        print("\n===== END DEBUG MODE =====")
        
    except Exception as e:
        print(f"\nError in debug_models: {e}")
        import traceback
        traceback.print_exc()

def debug_state_machine():
    """Debug state machine transitions and timings."""
    try:
        print("\n===== STATE MACHINE DEBUG MODE =====")
        
        # Create cycle manager for testing
        manager = PredictionCycleManager()
        
        # Test state transitions
        print("\nTesting state transitions:")
        initial_state = manager.state
        print(f"Initial state: {initial_state.value}")
        
        # Test each state handler method manually
        print("\nTesting FETCHING state handler:")
        manager.state = PredictionState.FETCHING
        manager._handle_fetching_state()
        print(f"After fetching handler, state is: {manager.state.value}")
        
        print("\nTesting ANALYZING state handler:")
        manager.state = PredictionState.ANALYZING
        manager._handle_analyzing_state()
        print(f"After analyzing handler, state is: {manager.state.value}")
        
        print("\nTesting PREDICTING state handler:")
        manager.state = PredictionState.PREDICTING
        manager._handle_predicting_state()
        print(f"After predicting handler, state is: {manager.state.value}")
        
        print("\nTesting WAITING state handler:")
        manager.state = PredictionState.WAITING
        # We'll just check the logic, not actually execute the waiting
        print(f"Waiting state handler would wait until {manager.scheduler.get_evaluation_time(manager.scheduler.get_next_draw_time()).strftime('%H:%M:%S')}")
        
        print("\nTesting scheduler methods:")
        next_draw = manager.scheduler.get_next_draw_time()
        evaluation_time = manager.scheduler.get_evaluation_time(next_draw)
        print(f"Next draw time: {next_draw.strftime('%H:%M:%S')}")
        print(f"Next evaluation time: {evaluation_time.strftime('%H:%M:%S')}")
        
        print("\n===== END STATE MACHINE DEBUG =====")
        
    except Exception as e:
        print(f"\nError in debug_state_machine: {e}")
        import traceback
        traceback.print_exc()

def run_automation(args):
    """Run the main automation cycle with provided arguments."""
    display_header()
    
    # Check debug modes first
    if args.debug_models:
        debug_models()
        return
    
    # Then check test mode    
    if args.test:
        print("Running in TEST MODE - automation cycle will not start.")
        print("Checking configuration and imports...")
        
        try:
            test_operation()
            
            # Ensure models are trained
            if not ensure_models_trained():
                print("Warning: Model check/training failed, but continuing test...")
            
            # Test state machine debug features
            debug_state_machine()
            
            # Test data collection
            collector = KinoDataCollector()
            print("\nTesting data collection...")
            draws = collector.fetch_latest_draws(num_draws=1)
            if draws:
                print("Data collection test successful!")
            
            print("\nAll tests passed. Run without --test flag to start the automation cycle.")
        except Exception as e:
            print(f"Test failed with error: {str(e)}")
            traceback.print_exc()
        return
    
    # Normal operation mode
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)
        
        # Ensure models are properly trained
        if not ensure_models_trained():
            print("\nFailed to ensure models are trained. Exiting.")
            return
        
        # Configure the cycle manager
        manager = PredictionCycleManager()
        manager.max_failures = args.max_failures
        manager.retry_delay = args.retry_delay
        manager.scheduler.post_draw_wait_seconds = args.post_draw_wait
        manager.fetch_retries = args.fetch_retries
        
        # Enable continuous learning if requested
        manager.continuous_learning_enabled = args.enable_learning
        if args.enable_learning:
            print("\nContinuous learning is ENABLED")
        
        # Set initial state if specified
        if args.initial_state:
            try:
                manager.state = PredictionState(args.initial_state)
                print(f"\nStarting with initial state: {manager.state.value}")
            except ValueError:
                print(f"\nWarning: Invalid initial state '{args.initial_state}'. Using default.")
        
        # Use Europe/Bucharest timezone (UTC+2)
        current_time = datetime.now(pytz.timezone('Europe/Bucharest'))
        print(f"\nStarting automation with configuration (UTC+2 time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}):")
        print(f"- Maximum consecutive failures: {manager.max_failures}")
        print(f"- Retry delay: {manager.retry_delay} seconds")
        print(f"- Post-draw wait: {manager.scheduler.post_draw_wait_seconds} seconds")
        print(f"- Draw interval: {manager.scheduler.draw_interval_minutes} minutes")
        print(f"- Fetch retries: {manager.fetch_retries}")
        print(f"- Continuous learning: {'Enabled' if manager.continuous_learning_enabled else 'Disabled'}")
        print(f"- Initial state: {manager.state.value}")
        
        print("\nStarting automation cycle...\n")
        manager.run_cycle()
        
    except KeyboardInterrupt:
        print("\nAutomation stopped by user.")
    except Exception as e:
        print(f"\nCritical error occurred: {str(e)}")
        traceback.print_exc()
        print("\nAutomation stopped due to error.")
    finally:
        print("\n" + "="*50)
        print("      AUTOMATION SYSTEM SHUTDOWN")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    run_automation(args)
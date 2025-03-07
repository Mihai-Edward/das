# File: automation/automation_runner.py
import os
import sys
import argparse
import time
import signal
import logging
from datetime import datetime
import traceback
import pytz

# Add src directory directly to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)  # Add src directory directly
sys.path.insert(0, project_root)  # Also add project root

# Now import core components from src
from config.paths import PATHS, ensure_directories
from draw_handler import DrawHandler, perform_complete_analysis, train_and_predict
from data_collector_selenium import KinoDataCollector
from prediction_evaluator import PredictionEvaluator

# Import automation components using relative imports
from .cycle_manager import PredictionCycleManager
from .scheduler import DrawScheduler, get_formatted_time_remaining
from .operations import test_operation  # Using relative import for operations
def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Automated lottery prediction system runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode without starting the cycle'
    )
    
    return parser


def display_header():
    """Display welcome message and system information."""
    print("\n" + "="*50)
    print("      AUTOMATED LOTTERY PREDICTION SYSTEM")
    print("="*50)
    
    now = datetime.now()
    scheduler = DrawScheduler()
    next_draw = scheduler.get_next_draw_time(now)
    eval_time = scheduler.get_evaluation_time(next_draw)
    
    print(f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Next draw at: {next_draw.strftime('%H:%M:%S')}")
    print(f"Next evaluation at: {eval_time.strftime('%H:%M:%S')}")
    print(f"Time until next draw: {get_formatted_time_remaining(next_draw)}")
    print(f"System directory: {current_dir}")
    
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
    print("\nReceived shutdown signal. Cleaning up...")
    print("Please wait for current operation to complete...")
    # Signal the cycle manager to stop gracefully
    global should_run
    should_run = False


def run_automation(args):
    """Run the main automation cycle with provided arguments."""
    display_header()
    
    if args.test:
        print("Running in TEST MODE - automation cycle will not start.")
        print("Checking configuration and imports...")
        
        try:
            from .operations import test_operation
            test_operation()
            
            manager = PredictionCycleManager()
            status = manager.get_status()
            
            print("\nTest successful! Components loaded correctly.")
            print(f"Next draw calculations working: {manager.scheduler.get_next_draw_time().strftime('%H:%M:%S')}")
            
            # Test fetch operation
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
    
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)
        
        # Configure and start the cycle manager
        manager = PredictionCycleManager()
        manager.max_failures = args.max_failures
        manager.retry_delay = args.retry_delay
        manager.scheduler.post_draw_wait_seconds = args.post_draw_wait
        # Add UTC time handling
        current_utc = datetime.now(pytz.UTC)
        print(f"\nStarting automation with configuration (UTC time: {current_utc.strftime('%Y-%m-%d %H:%M:%S')}):")
        print(f"- Maximum consecutive failures: {manager.max_failures}")
        print(f"- Retry delay: {manager.retry_delay} seconds")
        print(f"- Post-draw wait: {manager.scheduler.post_draw_wait_seconds} seconds")
        print(f"- Draw interval: {manager.scheduler.draw_interval_minutes} minutes")
        print(f"Starting automation with configuration:")
        print(f"- Maximum consecutive failures: {manager.max_failures}")
        print(f"- Retry delay: {manager.retry_delay} seconds")
        print(f"- Post-draw wait: {manager.scheduler.post_draw_wait_seconds} seconds")
        print(f"- Draw interval: {manager.scheduler.draw_interval_minutes} minutes")
        print(f"- Fetch retries: {args.fetch_retries}")
        
        # Start the automation cycle
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
    should_run = True  # Global flag for graceful shutdown
    parser = setup_parser()
    args = parser.parse_args()
    run_automation(args)
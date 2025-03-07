# File: automation/automation_runner.py
import os
import sys
import argparse
import time
import signal
import logging
from datetime import datetime
import traceback

# Add src directory directly to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)  # Add src directory directly
sys.path.insert(0, project_root)  # Also add project root

# Now the imports should work correctly
from config.paths import PATHS, ensure_directories
from draw_handler import DrawHandler, perform_complete_analysis, train_and_predict
from data_collector_selenium import KinoDataCollector
from prediction_evaluator import PredictionEvaluator

# Import components
from automation.cycle_manager import PredictionCycleManager

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
    
    print(f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System directory: {current_dir}")
    
    # Check if required directories exist
    try:
        ensure_directories()
        print("Directory structure verified.")
    except Exception as e:
        print(f"Warning: Issue with directories - {str(e)}")
    
    print("\nPress Ctrl+C at any time to stop the automation.")
    print("="*50 + "\n")


def run_automation(args):
    """Run the main automation cycle with provided arguments."""
    display_header()
    
    if args.test:
        print("Running in TEST MODE - automation cycle will not start.")
        print("Checking configuration and imports...")
        
        # Test import all required components
        try:
            # CHANGE THIS LINE - use correct import approach
            # Import directly since we already added project_root to sys.path
            from operations import test_operation  # Changed from automation.operations
            test_operation()
            
            manager = PredictionCycleManager()
            status = manager.get_status()
            
            print("\nTest successful! Components loaded correctly.")
            print(f"Next draw calculations working: {manager.scheduler.get_next_draw_time().strftime('%H:%M:%S')}")
            print("Run without --test flag to start the automation cycle.")
        except Exception as e:
            print(f"Test failed with error: {str(e)}")
            traceback.print_exc()
        return
    
    try:
        # Configure and start the cycle manager
        manager = PredictionCycleManager()
        manager.max_failures = args.max_failures
        manager.retry_delay = args.retry_delay
        manager.scheduler.post_draw_wait_seconds = args.post_draw_wait
        
        print(f"Starting automation with configuration:")
        print(f"- Maximum consecutive failures: {manager.max_failures}")
        print(f"- Retry delay: {manager.retry_delay} seconds")
        print(f"- Post-draw wait: {manager.scheduler.post_draw_wait_seconds} seconds")
        print(f"- Draw interval: {manager.scheduler.draw_interval_minutes} minutes")
        
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
    parser = setup_parser()
    args = parser.parse_args()
    run_automation(args)
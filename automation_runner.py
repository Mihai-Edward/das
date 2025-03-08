import glob
import os
import sys
import signal
import traceback
from datetime import datetime
import pytz
import argparse
import logging
import asyncio

# Add project root and src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from automation.cycle_manager import CycleManager
from automation.scheduler import DrawScheduler
from automation.operations import Operations
from src.lottery_predictor import LotteryPredictor
from config.paths import ensure_directories, PATHS

# Default configuration
DEFAULT_CONFIG = {
    'max_failures': 3,
    'retry_delay': 20,
    'enable_learning': True,
    'post_draw_wait': 50,
    'fetch_retries': 3
}

def init_logging():
    """Initialize logging configuration"""
    log_dir = os.path.join(project_root, 'data', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    current_time = datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'automation_{current_time}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('Automation')

def display_header():
    """Display welcome message and system information"""
    print("\n" + "="*50)
    print("      AUTOMATED LOTTERY PREDICTION SYSTEM")
    print("="*50)
    
    # Get current time in UTC and Bucharest time
    utc_now = datetime.now(pytz.UTC)
    bucharest_tz = pytz.timezone('Europe/Bucharest')
    bucharest_now = utc_now.astimezone(bucharest_tz)
    
    print(f"\nCurrent UTC time: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current Bucharest time: {bucharest_now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User: {os.getenv('USERNAME', 'Unknown')}")
    
    scheduler = DrawScheduler()
    next_draw = scheduler.get_next_draw_time()
    eval_time = scheduler.get_evaluation_time(next_draw)
    
    print(f"\nNext draw at: {next_draw.strftime('%H:%M:%S')}")
    print(f"Next evaluation at: {eval_time.strftime('%H:%M:%S')}")
    print(f"System directory: {os.path.dirname(__file__)}")
    
    print("\nRunning with configuration:")
    print(f"- Maximum failures: {DEFAULT_CONFIG['max_failures']}")
    print(f"- Retry delay: {DEFAULT_CONFIG['retry_delay']} seconds")
    print(f"- Post-draw wait: {DEFAULT_CONFIG['post_draw_wait']} seconds")
    print(f"- Fetch retries: {DEFAULT_CONFIG['fetch_retries']}")
    print(f"- Continuous learning: Enabled")
    
    print("\nPress Ctrl+C at any time to stop the automation.")
    print("="*50 + "\n")

def graceful_shutdown(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n\nReceived shutdown signal. Cleaning up...")
    print("Please wait for current operation to complete...")
    
    try:
        # Log shutdown event
        logger = logging.getLogger('Shutdown')
        logger.info("Graceful shutdown initiated")
        
        # Clean up temporary files
        temp_files = [
            os.path.join(PATHS['PROCESSED_DIR'], '*.temp'),
            os.path.join(PATHS['MODELS_DIR'], '*.temp')
        ]
        for pattern in temp_files:
            try:
                for f in glob.glob(pattern):
                    os.remove(f)
            except Exception:
                pass
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        sys.exit(0)

async def initialize_models():
    """Initialize and train models if needed"""
    logger = logging.getLogger('ModelInitialization')
    logger.info("Starting model initialization...")
    
    try:
        predictor = LotteryPredictor()
        is_valid, message = predictor.validate_model_state()
        
        if is_valid:
            logger.info("✓ Models already initialized and valid")
            return True
            
        logger.info("Models need initialization. Starting training process...")
        
        # Load historical data
        historical_data = predictor.load_data()
        if historical_data is None or len(historical_data) < 6:
            logger.error("Insufficient historical data for training")
            return False
            
        logger.info(f"Loaded {len(historical_data)} historical draws for training")
        
        try:
            # Prepare data for training
            logger.info("Preparing training data...")
            features, labels = predictor.prepare_data(historical_data)
            
            if features is None or labels is None:
                logger.error("Failed to prepare training data - features or labels are None")
                return False
                
            if len(features) == 0 or len(labels) == 0:
                logger.error("Failed to prepare training data - empty features or labels")
                return False
                
            logger.info(f"Training data prepared successfully:")
            logger.info(f"- Features shape: {features.shape}")
            logger.info(f"- Labels shape: {labels.shape}")
            
            # Train models with the prepared data
            logger.info("Starting model training...")
            training_success = predictor.train_models(features, labels)  # Pass both features and labels
            
            if training_success:
                logger.info("✓ Models trained successfully")
                
                # Validate the newly trained models
                is_valid, message = predictor.validate_model_state()
                if is_valid:
                    logger.info("✓ Model validation passed after training")
                    return True
                else:
                    logger.error(f"Model validation failed after training: {message}")
                    return False
            else:
                logger.error("Model training failed")
                return False
                
        except ValueError as ve:
            logger.error(f"Data preparation or training error: {str(ve)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during model training: {str(e)}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        traceback.print_exc()
        return False

async def start_automation():
    """Start the automation system with initialized models"""
    logger = logging.getLogger('AutomationStarter')
    
    try:
        # Initialize models first
        if not await initialize_models():
            logger.error("Failed to initialize models. Cannot start automation.")
            return False
            
        # Create and configure cycle manager with default configuration
        manager = CycleManager(
            max_failures=DEFAULT_CONFIG['max_failures'],
            retry_delay=DEFAULT_CONFIG['retry_delay'],
            enable_learning=DEFAULT_CONFIG['enable_learning']
        )
        
        # Set up operations with configuration
        ops = Operations()
        validation = ops.validate_system_state()
        if not validation.success:
            logger.error(f"System validation failed: {validation.error}")
            return False
            
        logger.info("Starting automation cycle...")
        await manager.run_cycle()
        return True
        
    except Exception as e:
        logger.error(f"Error starting automation: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main entry point for automation"""
    # Initialize logging
    logger = init_logging()
    logger.info("Starting automation system...")
    
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)
        
        # Display welcome header
        display_header()
        
        # Ensure directories exist
        if not ensure_directories():
            logger.error("Failed to create required directories")
            return
        
        # Start the main automation with default configuration
        asyncio.run(start_automation())
        
    except KeyboardInterrupt:
        logger.info("\nAutomation stopped by user")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        traceback.print_exc()
    finally:
        logger.info("Automation system shutdown complete")

if __name__ == "__main__":
    main()
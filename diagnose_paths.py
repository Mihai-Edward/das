import os
import sys
import importlib
import platform
from datetime import datetime
import pytz
import logging

# Fix the path - add src directory directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

def check_import(module_name):
    """Try to import a module and report its status."""
    print(f"\nTrying to import: {module_name}")
    try:
        module = importlib.import_module(module_name)
        print(f"✓ Successfully imported {module_name}")
        print(f"  Module file location: {module.__file__}")
        return True, module
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False, None

def check_environment():
    """Check and display environment information."""
    print("\nEnvironment Information:")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check timezone configuration
    bucharest_tz = pytz.timezone('Europe/Bucharest')
    utc_now = datetime.now(pytz.UTC)
    bucharest_now = utc_now.astimezone(bucharest_tz)
    
    print(f"Current UTC Time: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current Bucharest Time: {bucharest_now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Project Root: {project_root}")
    print(f"Source Directory: {src_dir}")

def check_automation_components():
    """Check automation-specific components and their configurations."""
    print("\nAutomation Components Check:")
    
    # Check scheduler
    try:
        from automation.scheduler import DrawScheduler
        scheduler = DrawScheduler()
        next_draw = scheduler.get_next_draw_time()
        eval_time = scheduler.get_evaluation_time(next_draw)
        print("✓ Scheduler initialized successfully")
        print(f"  Next draw time: {next_draw.strftime('%H:%M:%S')}")
        print(f"  Next evaluation time: {eval_time.strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"✗ Scheduler check failed: {e}")
    
    # Check operations
    try:
        from automation.operations import Operations
        ops = Operations()
        print("✓ Operations initialized successfully")
        validation = ops.validate_system_state()
        if validation.success:
            print("✓ System state validation passed")
            for key, value in validation.data.items():
                print(f"  - {key}: {value}")
        else:
            print(f"✗ System state validation failed: {validation.error}")
    except Exception as e:
        print(f"✗ Operations check failed: {e}")
    
    # Check cycle manager
    try:
        from automation.cycle_manager import CycleManager, PredictionState
        manager = CycleManager()
        print("✓ Cycle Manager initialized successfully")
        print(f"  Current state: {manager.state.value}")
        print(f"  States available: {[state.value for state in PredictionState]}")
    except Exception as e:
        print(f"✗ Cycle Manager check failed: {e}")

def check_timing_configuration():
    """Verify timing-related configurations and tolerances."""
    print("\nTiming Configuration Check:")
    
    try:
        from automation.scheduler import DrawScheduler
        scheduler = DrawScheduler()
        
        # Check basic timing parameters
        print(f"Draw interval: {scheduler.draw_interval_minutes} minutes")
        print(f"Post-draw wait: {scheduler.post_draw_wait_seconds} seconds")
        print(f"Timing tolerance: {scheduler.timing_tolerance} seconds")
        
        # Test time calculations
        current = scheduler.get_current_time()
        next_draw = scheduler.get_next_draw_time()
        eval_time = scheduler.get_evaluation_time(next_draw)
        
        print("\nTiming Calculations Test:")
        print(f"Current time: {current.strftime('%H:%M:%S')}")
        print(f"Next draw: {next_draw.strftime('%H:%M:%S')}")
        print(f"Next evaluation: {eval_time.strftime('%H:%M:%S')}")
        
        # Verify tolerances
        time_to_draw = (next_draw - current).total_seconds()
        time_to_eval = (eval_time - current).total_seconds()
        
        print("\nTiming Windows:")
        print(f"Time until next draw: {time_to_draw:.1f} seconds")
        print(f"Time until evaluation: {time_to_eval:.1f} seconds")
        
    except Exception as e:
        print(f"✗ Timing configuration check failed: {e}")

def check_directories():
    """Check required project directories with automation focus."""
    dirs_to_check = [
        ('Project Root', project_root),
        ('Source Directory', src_dir),
        ('Config Directory', os.path.join(project_root, 'config')),
        ('Automation Directory', os.path.join(project_root, 'automation')),
        ('Data Directory', os.path.join(project_root, 'data')),
        ('Processed Data', os.path.join(project_root, 'data', 'processed')),
        ('Models Directory', os.path.join(project_root, 'models')),
        ('Drivers Directory', os.path.join(project_root, 'drivers'))
    ]
    
    print("\nDirectory Structure Check:")
    for name, dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"✓ {name}: {dir_path}")
            init_file = os.path.join(dir_path, '__init__.py')
            if name in ['Source Directory', 'Config Directory', 'Automation Directory']:
                if os.path.exists(init_file):
                    print(f"  ✓ __init__.py present")
                else:
                    print(f"  ✗ __init__.py missing")
        else:
            print(f"✗ {name} missing: {dir_path}")

def check_automation_files():
    """Check automation-specific files."""
    files_to_check = [
        ('Scheduler', os.path.join(project_root, 'automation', 'scheduler.py')),
        ('Operations', os.path.join(project_root, 'automation', 'operations.py')),
        ('Cycle Manager', os.path.join(project_root, 'automation', 'cycle_manager.py')),
        ('Automation Runner', os.path.join(project_root, 'automation', 'automation_runner.py')),
        ('Paths Config', os.path.join(project_root, 'config', 'paths.py'))
    ]
    
    print("\nAutomation Files Check:")
    for name, file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {name} present: {file_path}")
            # Check file modification time
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"  Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"✗ {name} missing: {file_path}")

def check_critical_files():
    """Check presence of critical project files."""
    files_to_check = [
        ('Edge Driver', os.path.join(project_root, 'drivers', 'msedgedriver.exe')),
        ('Historical Data', os.path.join(src_dir, 'historical_draws.csv')),
        ('Paths Config', os.path.join(project_root, 'config', 'paths.py')),
        ('Main Script', os.path.join(src_dir, 'main.py')),
        ('Draw Handler', os.path.join(src_dir, 'draw_handler.py')),
        ('Data Collector', os.path.join(src_dir, 'data_collector_selenium.py')),
        ('Predictor', os.path.join(src_dir, 'lottery_predictor.py'))
    ]
    
    print("\nCritical Files Check:")
    for name, file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {name} present: {file_path}")
        else:
            print(f"✗ {name} missing: {file_path}")

def check_imports():
    """Check all required module imports."""
    imports_to_check = [
        'config',
        'config.paths',
        'draw_handler',
        'lottery_predictor',
        'data_analysis',
        'data_collector_selenium',
        'prediction_evaluator',
        'automation.cycle_manager',
        'automation.scheduler',
        'automation.operations'
    ]
    
    print("\nModule Import Check:")
    for module_name in imports_to_check:
        success, module = check_import(module_name)
        if success and module_name == 'config.paths':
            if hasattr(module, 'PATHS'):
                print("  ✓ PATHS configuration found")
                print(f"  Configuration keys: {list(module.PATHS.keys())}")
            else:
                print("  ✗ PATHS configuration missing")

def verify_selenium_setup():
    """Verify Selenium and browser driver setup."""
    try:
        from selenium import webdriver
        from selenium.webdriver.edge.service import Service
        print("\nSelenium Setup Check:")
        print("✓ Selenium package installed")
        
        driver_path = os.path.join(project_root, 'drivers', 'msedgedriver.exe')
        if os.path.exists(driver_path):
            print(f"✓ Edge driver found: {driver_path}")
            
            # Test driver initialization
            try:
                service = Service(executable_path=driver_path)
                options = webdriver.EdgeOptions()
                options.add_argument('--headless')
                driver = webdriver.Edge(service=service, options=options)
                driver.quit()
                print("✓ Edge driver initialization successful")
            except Exception as e:
                print(f"✗ Edge driver initialization failed: {e}")
        else:
            print(f"✗ Edge driver missing: {driver_path}")
            
    except ImportError as e:
        print(f"✗ Selenium setup incomplete: {e}")

def run_diagnostics():
    """Run all diagnostic checks with focus on automation system."""
    try:
        print("="*50)
        print("Running Project Diagnostics (Automation Focus)")
        print(f"Timestamp: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*50)
        
        check_environment()
        check_directories()
        check_automation_files()
        check_automation_components()
        check_timing_configuration()
        check_critical_files()
        check_imports()
        verify_selenium_setup()
        
        print("\n" + "="*50)
        print("Diagnostics Complete")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during diagnostics: {e}")
        raise

if __name__ == "__main__":
    run_diagnostics()
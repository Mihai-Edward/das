import os
import sys
import importlib
import platform
from datetime import datetime
import pytz

# Fix the path - add src directory directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

def check_import(module_name):
    """
    Try to import a module and report its status.
    
    Args:
        module_name (str): Name of module to import
        
    Returns:
        tuple: (success, module)
            - success (bool): True if import successful
            - module: The imported module if successful, None otherwise
    """
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
    print(f"Current UTC Time: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Project Root: {project_root}")
    print(f"Source Directory: {src_dir}")

def check_paths():
    """Check Python path configuration."""
    print("\nPython Path Configuration:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")

def check_directories():
    """Check required project directories."""
    dirs_to_check = [
        ('Project Root', project_root),
        ('Source Directory', src_dir),
        ('Config Directory', os.path.join(project_root, 'config')),
        ('Automation Directory', os.path.join(project_root, 'automation')),
        ('Data Directory', os.path.join(project_root, 'data')),
        ('Models Directory', os.path.join(project_root, 'models')),
        ('Drivers Directory', os.path.join(project_root, 'drivers'))
    ]
    
    print("\nDirectory Structure Check:")
    for name, dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"✓ {name}: {dir_path}")
            init_file = os.path.join(dir_path, '__init__.py')
            if os.path.exists(init_file):
                print(f"  ✓ __init__.py present")
            else:
                print(f"  ✗ __init__.py missing")
        else:
            print(f"✗ {name} missing: {dir_path}")

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
    """Run all diagnostic checks."""
    try:
        print("="*50)
        print("Running Project Diagnostics")
        print(f"Timestamp: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*50)
        
        check_environment()
        check_paths()
        check_directories()
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
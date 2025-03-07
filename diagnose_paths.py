# File: automation/diagnose_paths.py
import os
import sys
import importlib

# Fix the path - add src directory directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level to project root
src_dir = os.path.join(project_root, 'src')  # Get path to src directory
sys.path.insert(0, src_dir)  # Add src directory directly to path
sys.path.insert(0, project_root)  # Add project root to beginning of path

def check_import(module_name):
    print(f"\nTrying to import: {module_name}")
    try:
        module = importlib.import_module(module_name)
        print(f"✓ Successfully imported {module_name}")
        print(f"  Module file location: {module.__file__}")
        return True, module
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False, None

def main():
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Project root: {project_root}")
    print(f"Src directory: {src_dir}")
    
    print("\nPython path after correction:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    # Directory checks - updated to include src directory
    dirs_to_check = [
        os.path.join(project_root, 'config'),
        os.path.join(project_root, 'automation'),
        os.path.join(project_root, 'src')  # Added src directory check
    ]
    
    print("\nDirectory existence check:")
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists")
            init_file = os.path.join(dir_path, '__init__.py')
            if os.path.exists(init_file):
                print(f"  ✓ {dir_path}/__init__.py exists")
            else:
                print(f"  ✗ {dir_path}/__init__.py missing")
        else:
            print(f"✗ {dir_path} doesn't exist")
    
    # Try different import approaches - updated to import directly
    imports_to_try = [
        'config',
        'config.paths',
        'draw_handler',       # Direct import from src
        'lottery_predictor',  # Direct import from src
        'data_analysis',      # Direct import from src
        'data_collector_selenium',  # Direct import from src
        'prediction_evaluator',     # Direct import from src
        'automation',
        'automation.cycle_manager'
    ]
    
    for import_name in imports_to_try:
        success, module = check_import(import_name)
        if success and import_name == 'config.paths':
            if hasattr(module, 'PATHS'):
                print("  ✓ PATHS variable found in config.paths")
                print(f"  Keys in PATHS: {list(module.PATHS.keys())}")
            else:
                print("  ✗ PATHS variable NOT found in config.paths")
        
if __name__ == "__main__":
    main()
import os

# Base directory (root of the project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths relative to BASE_DIR
PATHS = {
    'HISTORICAL_DATA': os.path.join(BASE_DIR, "src", "historical_draws.csv"),
    'PREDICTIONS': os.path.join(BASE_DIR, "data", "processed", "predictions.csv"),
    'ANALYSIS': os.path.join(BASE_DIR, "data", "processed", "analysis_results.xlsx"),
    'MODELS_DIR': os.path.join(BASE_DIR, "models"),  # Changed from src/ml_models to models/
    'DRIVER': os.path.join(BASE_DIR, "drivers", "msedgedriver.exe"),
}

def ensure_directories():
    """Ensure all required directories exist with proper structure"""
    try:
        # First, create the main directories
        required_dirs = [
            os.path.join(BASE_DIR, "data", "processed"),
            os.path.join(BASE_DIR, "models"),  # Changed from src/ml_models
            os.path.join(BASE_DIR, "drivers"),
            os.path.join(BASE_DIR, "src")
        ]
        
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured directory exists: {directory}")
        
        # Then ensure all path directories exist
        for name, path in PATHS.items():
            directory = os.path.dirname(path)
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured path exists for {name}: {directory}")
            
        # Create models directory explicitly
        os.makedirs(PATHS['MODELS_DIR'], exist_ok=True)
        
        # Create a timestamp file directory
        timestamp_dir = os.path.dirname(PATHS['MODELS_DIR'])
        os.makedirs(timestamp_dir, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False

def validate_paths():
    """Validate that all required paths exist and are accessible"""
    missing_paths = []
    inaccessible_paths = []
    
    for name, path in PATHS.items():
        dir_path = os.path.dirname(path)
        
        # Check if directory exists
        if not os.path.exists(dir_path):
            missing_paths.append(f"{name}: {dir_path}")
            continue
            
        # Check if directory is writable
        try:
            test_file = os.path.join(dir_path, '.test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except (IOError, OSError):
            inaccessible_paths.append(f"{name}: {dir_path}")
    
    if missing_paths or inaccessible_paths:
        if missing_paths:
            print("\nThe following paths are missing:")
            for path in missing_paths:
                print(f"- {path}")
        if inaccessible_paths:
            print("\nThe following paths are not writable:")
            for path in inaccessible_paths:
                print(f"- {path}")
        return False
        
    return True

def get_absolute_path(path_key):
    """Get the absolute path for a given path key"""
    if path_key not in PATHS:
        raise KeyError(f"Unknown path key: {path_key}")
    return os.path.abspath(PATHS[path_key])

# Validate paths when module is loaded
if __name__ == "__main__":
    print(f"Base Directory: {BASE_DIR}")
    ensure_directories()
    if validate_paths():
        print("\nAll paths are valid and accessible")
    else:
        print("\nSome paths have issues. Please check the warnings above")
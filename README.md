# DAS (Data Analysis System) for Lottery Prediction

## Project Overview
An advanced machine learning-based system for lottery number prediction and analysis, featuring real-time data collection, statistical analysis, and ML-driven predictions.

## Features
- 🤖 Machine Learning Prediction Engine
- 📊 Statistical Analysis & Pattern Recognition
- 🌐 Automated Web Data Collection
- 📈 Performance Evaluation & Tracking
- 💾 Historical Data Management
- 🔄 Automated Pipeline Testing

## Project Structure
```
├── src/                           # Source code directory
│   ├── __init__.py               # Package initializer
│   ├── main.py                   # Main application entry point
│   ├── lottery_predictor.py      # ML prediction engine
│   ├── data_analysis.py          # Statistical analysis tools
│   ├── data_collector_selenium.py # Web scraping system
│   ├── draw_handler.py           # Draw management system
│   ├── prediction_evaluator.py   # Results evaluation
│   └── historical_draws.csv      # Historical data storage
├── drivers/                       # WebDriver directory
│   └── msedgedriver.exe          # Microsoft Edge WebDriver
├── data/                         # Data storage directory
│   └── processed/                # Processed data storage
│       ├── predictions.csv       # Generated predictions
│       └── analysis_results.xlsx # Analysis results
└── config/                       # Configuration directory
    ├── __init__.py              # Config package initializer
    └── paths.py                 # Path configurations
```

## Core Components

### 1. ML Prediction Engine (`lottery_predictor.py`)
- Probabilistic and pattern-based models
- Feature engineering and data preprocessing
- Model training and evaluation

### 2. Data Collection (`data_collector_selenium.py`)
- Automated web scraping
- Real-time data validation
- Historical data management

### 3. Statistical Analysis (`data_analysis.py`)
- Frequency analysis
- Pattern recognition
- Hot/cold number analysis
- Sequence pattern detection

### 4. Draw Management (`draw_handler.py`)
- Draw data handling
- Model pipeline coordination
- Result persistence

### 5. Prediction Evaluation (`prediction_evaluator.py`)
- Accuracy tracking
- Performance metrics
- Trend analysis

## Installation

### Prerequisites
- Python 3.8+
- Microsoft Edge WebDriver
- Required Python packages:
```bash
pip install pandas numpy scikit-learn selenium beautifulsoup4 matplotlib seaborn
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Mihai-Edward/das.git
cd das
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure paths in `config/paths.py`

## Usage

Run the main application:
```bash
python src/main.py
```

Available options:
- 3: Fetch latest draws
- 8: Perform complete analysis
- 9: Generate ML prediction
- 10: Evaluate prediction accuracy
- 11: Run pipeline test
- 12: Exit

## License
[Your chosen license]

## Author
Mihai-Edward
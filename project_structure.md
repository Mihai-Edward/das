project name
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── draw_handler.py
│   ├── lottery_predictor.py
│   ├── data_analysis.py
│   ├── data_collector_selenium.py
│   └── historical_draws.csv
├── drivers/
│   └── msedgedriver.exe
├── data/
│   └── processed/
│       ├── evaluation_results.xlsx
│       ├── performance_trends.png
│       ├── predictions.csv
│       └── analysis_results.xlsx
├── config/
│   ├── __init__.py
│   └── paths.py
└── automation/
    ├── __init__.py
    ├── scheduler.py        # Time calculations for draws
    ├── operations.py       # Non-interactive menu options
    ├── cycle_manager.py    # Core automation logic
    ├── automation_runner.py # Command-line entry point
    └── diagnose_paths.py   # File to diagnose paths
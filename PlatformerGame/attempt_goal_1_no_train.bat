@echo off
cd %~dp0
python trigger_training.py --location "%~1" --goal_level "Data\Test\goal1.json" --training_iterations 0 --diamonds_medal_thresholds 5 10 15 --levels_medal_threshold 3 5 7
pause

@echo off
cd %~dp0
python trigger_training.py --location "%~1" --training_iterations 400 --goal_level "Data\Test\Tutorial\goal1.json" --diamonds_medal_thresholds 3 5 7 --levels_medal_threshold 2 3 4
pause
rem Copyright (c) Microsoft Corporation.
rem Licensed under the MIT License.

@echo off
cd %~dp0
python trigger_training.py --location "%~1" --goal_level "Data\Test\goal2.json" --training_iterations 800 --diamonds_medal_thresholds 5 10 15 --levels_medal_threshold 3 5 7
pause
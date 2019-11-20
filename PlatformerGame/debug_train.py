# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import trigger_training

def main():
    parser = trigger_training.create_arg_parser()
    args = parser.parse_args(['--location', r'.\Demo\complete', '--training_iterations', '800', '--goal_level', r'.\Data\Test\goal1.json', '--diamonds_medal_thresholds', '5', '10', '15', '--levels_medal_threshold', '4', '6', '12'])
    trigger_training.main(args)

if __name__ == '__main__':
    main()
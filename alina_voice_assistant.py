#!/usr/bin/env python3

import argparse
import os

from modules.labels import save_labels
from modules.model import train


def parse_cli_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-model', type=str, help="Model file path")
    parser.add_argument('--save-labels', type=str, help="File path to put labels")
    arguments = parser.parse_args()
    return arguments


def main():
    arguments = parse_cli_arguments()
    if arguments.train_model:
        model_file_path = arguments.train_model
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f'model file {model_file_path} was not found')
        train(model_file_path)
    if arguments.save_labels:
        file_path = arguments.save_labels
        save_labels(file_path)


if __name__ == '__main__':
    main()

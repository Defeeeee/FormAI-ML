# FōrmAI Machine Learning Module

[![Three Clause BSD](https://img.shields.io/badge/License-BSD-green.svg)](https://opensource.org/license/bsd-3-clause) [![Commit Check](https://github.com/commit-check/commit-check-action/actions/workflows/commit-check.yml/badge.svg)](https://github.com/Defeeeee/FormAI/actions/workflows/commit-check.yml)   


<img src="https://github.com/user-attachments/assets/00931eda-1efb-4da2-be12-f5f7ab0af75d" width="250">


<br>

<br>

This repository contains the core machine learning components for the FōrmAI project, responsible for analyzing user movements and providing feedback on exercise technique.

## Project Structure

* **API:** Contains the FastAPI API (`main.py`) for interacting with the ML models.
* **Computer Vision:** Houses MediaPipe experiments and utilities for joint tracking and pose estimation.

## Tech Stack

* Python 3
* MediaPipe
* TensorFlow
* Numpy
* FastAPI

## Integration

This module is designed to be integrated as a submodule into the main FōrmAI project. The primary interaction will be through the API exposed in the `API` folder.

## Authors

* [@Defeeeee](https://github.com/Defeeeee) - Federico Diaz Nemeth

## License

FōrmAI © 2024 by Eric Gerzenstein, Federico Diaz Nemeth, Juan Baader and Dan Segal is licensed under the [BSD-3-Clause license](https://opensource.org/license/bsd-3-clause)
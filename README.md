# Autonomous Robot Cleaner


## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup Python Virtual Environment](#setup-python-virtual-environment)
- [Install Dependencies](#install-dependencies)
- [Running the Training Script](#running-the-training-script)
- [Running the Testing Script](#running-the-testing-script)

## Prerequisites

- Python 3.10 installed
- Requirements.txt intalled

## Setup Python Virtual Environment

Navigate to your project directory:
cd /path/to/your/project

Create a virtual environment:
python -m venv myenv

Replace myenv with the desired name of your environment.

Activate the virtual environment:

On Windows:
myenv\Scripts\activate

On macOS and Linux:
source myenv/bin/activate

Install Dependencies
pip install -r requirements.txt

Running the Training Script
python experiment.py

Running the Testing Script
python test_agent.py

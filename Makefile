.PHONY: download-data preprocess-data train evaluate

# Detect if we're on Windows and use appropriate Python command
ifeq ($(OS),Windows_NT)
	PYTHON = venv\Scripts\python.exe
else
	PYTHON = python3
endif

download-data:
	$(PYTHON) scripts/download_data.py

preprocess-data:
	$(PYTHON) -m data.preprocess

train:
	$(PYTHON) -m training.train_dqn

evaluate:
	$(PYTHON) -m evaluation.evaluate_dqn

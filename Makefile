.PHONY: download-data preprocess-data train evaluate

download-data:
	python3 scripts/download_data.py

preprocess-data:
	python3 -m data.preprocess

train:
	python3 -m training.train_dqn

evaluate:
	python3 -m evaluation.evaluate_dqn

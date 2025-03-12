# variables
DATA_FOLDER=data
DATASET=joebeachcapital/airbnb
NOTEBOOKS_FOLDER=notebooks
SRC_FOLDER=src
TESTS_FOLDER=tests
KAGGLE_CONFIG=/.kaggle/kaggle.json

# default target
all: setup setup-kaggle data pipeline test clean

# setup: install necessary dependencies
setup:
	@echo "setting up environment..."
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "environment setup complete!"
	@echo "To activate the virtual environment, run: source .venv/bin/activate"
	@echo "To deactivate the virtual environment, run: deactivate"

# setup the Kaggle API
setup-kaggle:
	@if [ ! -f $(KAGGLE_CONFIG) ]; then \
		echo "1. Login to your Kaggle account (or create an account if needed)."; \
		echo "2. Go to 'Settings > Account' > 'API' -> 'Create New Token'."; \
		echo "3. Save the downloaded 'kaggle.json' file to ~/.kaggle/ (you can create this folder by typing mkdir .kaggle from the repository"; \
		echo "4. Run: chmod 600 ~/.kaggle/kaggle.json in your terminal."; \
	fi
		
# download data from Kaggle
.PHONY: data
data:
	@echo "Downloading the dataset from Kaggle..."
	kaggle datasets download $(DATASET) -p $(DATA_FOLDER)
	unzip -o $(DATA_FOLDER)/*.zip -d $(DATA_FOLDER)
	rm -f $(DATA_FOLDER)/*.zip
	@echo "Data download complete."

#clean the data
clean-data:
	@echo "Cleaning the data..."
	.venv/bin/python3 src/clean_data.py
	@echo "Data cleaned and added to the data folder."

#run the models
.PHONY: models
models:
	@echo "Running the models..."
	.venv/bin/python3 src/models.py
	@echo "Models have been successfully run."

# run the data pipeline
pipeline:
	@echo "Starting the data pipeline..."
	jupyter nbconvert --to notebook --execute $(NOTEBOOKS_FOLDER)/data_pipeline.ipynb --output $(NOTEBOOKS_FOLDER)/data_pipeline_output.ipynb
	@echo "the data pipeline has been executed."

# run tests
test:
	@echo "Running tests..."
	pytest $(TESTS_FOLDER)
	@echo "tests complete."

# clean generated files
cleanup:
	@echo "Cleaning up..."
	rm -rf $(DATA_FOLDER)/*
	find . -name "__pycache__" -type d -exec rm -r {} +
	@echo "cleanup is complete!"

# help target to display valid commands
help:
	@echo "Available targets:"
	@echo "  setup        - Install dependencies"
	@echo "  setup-kaggle - Setup the Kaggle API"
	@echo "  data         - Download dataset from Kaggle"
	@echo "  clean-data   - Clean the data"
	@echo "  models       - Run the models"
	@echo "  pipeline     - Run the data pipeline"
	@echo "  test         - Run tests"
	@echo "  cleanup      - Remove generated files"
	@echo "  help         - Show this help message"

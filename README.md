# Basic MLE Project

The project is structured as follows:

- `data/`: Contains the training and inference datasets.
- `training/`: Contains the training script and Dockerfile.
- `inference/`: Contains the inference script and Dockerfile.
- `tests/`: Contains unit tests for data processing, model, and training.
- `data_processing.py`: Contains code to process Iris data.
- `.gitignore`: Filters out temporary files and models.
- `requirements.txt`: Contains all necessary dependencies.

## How to Run

1. **Data Processing**:
   - Run `python data_processing.py` to generate the training and inference datasets.

2. **Training**:
   - Build the Docker image: `docker build -t iris-train -f training/Dockerfile .`
   - Run the Docker container: `docker run --name train-container iris-train`
   - Copy model to local: `docker cp train-container:/app/model.pth ./model.pth`

3. **Inference**:
   - Build the Docker image: `docker build -t iris-inference -f inference/Dockerfile .`
   - Run the Docker container: `docker run iris-inference`

4. **Testing**:
   - Run `python -m unittest discover tests` to run all unit tests.
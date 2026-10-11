# ML Pipeline With FastAPI

Github repo: [Link](https://github.com/chrisestevez)

The project trains a Random Forest classification model on the UCI Census Income dataset to predict individual income levels. The model evaluates performance across specific demographic slices to explicitly check for algorithmic bias and disparities. Additionally the  resulting pipeline documents these findings through a structured Model Card framework to ensure transparency and responsible AI practices.

## Setup Instructions

### 1. Install Dependencies

Install [installation uv](https://docs.astral.sh/uv/getting-started/installation/)
* run `uv sync` to create and sync the `.venv`
* activate `.venv` source .venv/bin/activate on Linux/macOS or .venv\Scripts\activate on Windows

## Run Project
With uv installed.

### Train Model
```bash
uv run train_model.py 
```

### Run tests
```bash
uv run pytest test_ml.py 
```

### Run API
```bash
uv run uvicorn main:app --reload
```

### Run Sample Submission to data/ end point
```bash
uv run local_api.py
```
python3 -m venv ecommercePrediction
source ecommercePrediction/bin/activate
sales-prediction-mlops/
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── validation/
│
├── notebooks/
│ ├── data_exploration.ipynb
│ └── model_development.ipynb
│
├── src/
│ ├── data/
│ │ ├── **init**.py
│ │ ├── data_loader.py
│ │ └── data_preprocessor.py
│ │
│ ├── models/
│ │ ├── **init**.py
│ │ ├── train_model.py
│ │ └── predict_model.py
│ │
│ └── utils/
│ ├── **init**.py
│ └── logger.py
│
├── tests/
│ ├── test_data.py
│ └── test_model.py
│
├── configs/
│ ├── model_config.yaml
│ └── data_config.yaml
│
├── requirements.txt
├── setup.py
├── Dockerfile
├── .github/workflows/
│ └── ci_cd.yml
└── README.md

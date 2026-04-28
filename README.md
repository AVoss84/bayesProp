# Template for generic Python package (Data Science) 

This is a blueprint of a generic end-to-end data science project, i.e. building a Python package along the usual steps: data preprocessing, model training, prediction, postprocessing, REST API construction (for real-time model serving) and containerization for final deployment as a microservice.

## Package structure

```
├── CHANGELOG.md
├── Dockerfile_Fastapi       # vanilla rest api image
├── Dockerfile_Streamlit     # vanilla streamlit image
├── README.md
├── main.py                  # REST API definition 
├── build_run.sh             # for local containerization 
├── Makefile                 # run unit tests, linting etc.
├── data                     # local data dump
├── pyproject.toml 
├── src
│   ├── my_package
│   │   ├── config
│   │   │   ├── config.py
│   │   │   ├── global_config.py
│   │   │   ├── input_output.yaml
│   │   │   └── model_config.yaml
│   │   ├── resources
│   │   │   ├── postprocessor.py
│   │   │   ├── predictor.py
│   │   │   ├── preprocessor.py
│   │   │   └── trainer.py
│   │   ├── services
│   │   │   ├── file.py
│   │   │   ├── file_aws.py
│   │   │   ├── pipelines.py
│   │   │   └── publisher.py
│   │   └── utils
│   │       └── utils.py
│   └── notebooks
└── streamlit_app.py            # basic streamlit app
```

## Use Case description

**Business goal**: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 

**Business stakeholders**: Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

**Input data description**: Iris data set

**Business impact KPI**: Faster business process (in hours/days)


## Package installation and application develoment

Create virtual environment: 
```bash
# curl -LsSf https://astral.sh/uv/install.sh | sh           # optional: get uv manager
uv venv --python 3.12
uv sync
source .venv/bin/activate
```

Start REST API locally:
```bash
make api        # checkout Swagger docs: http://127.0.0.1:8000/docs 
```

Start Streamlit UI locally:
```bash
make ui       
```

Run formating: 
```bash
make format
```



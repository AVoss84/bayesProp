a.PHONY: format up down rm-all ui api free-port

LINT_FOLDER ?= src/my_package

# # For local streamlit server
# STREAMLIT_HOST ?= 127.0.0.1
# STREAMLIT_PORT ?= 5000

format:
	uv run mypy $(LINT_FOLDER)
	uv run black $(LINT_FOLDER)

# Start all services
up:
	docker compose up -d
	
down:
	docker compose down --volumes --remove-orphans

# Remove all containers, images, volumes, and networks
rm-all:
	docker-compose down --rmi all --volumes --remove-orphans

ui:
	streamlit run streamlit_app.py

api:
	uvicorn main:app  --reload

# free a specific port
free-port:
    lsof -t -i :$(1) | xargs kill -9

# make free-port PORT=5000

FROM python:3.12-slim
EXPOSE 8080
WORKDIR /app
COPY .streamlit /app/.streamlit
COPY streamlit_app.py /app/streamlit_app.py
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN pip install uv
RUN uv venv --python 3.12 && uv sync --no-cache

CMD [ "/bin/bash", "-c", "source .venv/bin/activate && streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0"]
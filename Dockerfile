FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory /app
RUN pip install poetry \
  && poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* ./
RUN poetry install

COPY ./robiemon_server ./robiemon_server

VOLUME ["/app/data"]
EXPOSE 3000
CMD ["poetry", "run", "task", "prod"]

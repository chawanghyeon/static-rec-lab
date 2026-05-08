FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --group dev --no-install-project

COPY . .
RUN uv sync --group dev

CMD ["uv", "run", "python", "-m", "apps.api.main"]

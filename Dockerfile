FROM python:3.10-slim

WORKDIR /app
# Install the exact framework requirements
RUN pip install --no-cache-dir openenv-core fastapi uvicorn pydantic openai

COPY . /app
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
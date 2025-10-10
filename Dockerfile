# Build stage
FROM python:3.10-slim as builder

WORKDIR /app
COPY ./requirements.prod.txt .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.prod.txt

# Production stage
FROM python:3.10-slim

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app/source/api
ENV PYTHONUNBUFFERED=1

COPY ./source /app/source

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
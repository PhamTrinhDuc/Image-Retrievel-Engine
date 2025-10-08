FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Copy only requirements first so this layer is cached unless requirements.txt changes
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source after installing deps so source changes won't bust the pip layer
COPY ./src /app/src

EXPOSE 8501
CMD ["python", "src/run_server.py"]
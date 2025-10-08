FROM python:3.10-slim

WORKDIR /app/source/api
ENV PYTHONUNBUFFERED=1

COPY ./requirements.prod.txt /app/requirements.prod.txt
RUN pip install --no-cache-dir -r /app/requirements.prod.txt

COPY ./source /app/source

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]

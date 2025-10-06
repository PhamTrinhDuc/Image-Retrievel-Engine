# Simple API endpoints for Image Retrieval System

This folder contains FastAPI endpoints for inference and testing.

## Files

- `route_search.py`: Main API endpoints with FastAPI
- `run_server.py`: Simple script to start the server  
- `test_client.py`: Client to test the API endpoints

## API Endpoints

### 1. Health Check
```
GET /health
```
Check if the API is running and models are loaded.

### 2. Search by Upload
```
POST /search/upload
```
Upload an image file and search for similar images.

**Parameters:**
- `file`: Image file (multipart/form-data)
- `top_k`: Number of results (default: 5)  
- `extractor_type`: Model type (default: "resnet")

### 3. Search by Base64
```
POST /search/base64
```
Search using base64 encoded image.

**Body:**
```json
{
  "image_base64": "base64_string",
  "top_k": 5,
  "extractor_type": "resnet"
}
```

### 4. Available Models
```
GET /models
```
Get list of available feature extraction models.

### 5. System Stats
```
GET /stats  
```
Get system statistics and information.

## Usage

### Start Server
```bash
cd source/api
python run_server.py

# Or with custom settings
python run_server.py --host 0.0.0.0 --port 8080
```

### Test API
```bash
python test_client.py
```

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## Available Extractors

- `resnet`: ResNet models (resnet34, resnet50, etc.)
- `vgg`: VGG models  
- `vit`: Vision Transformer
- `dinov2`: DINOv2 models
- `clip`: CLIP models

## Response Format

All search endpoints return:
```json
{
  "success": true,
  "results": [
    {
      "image_id": "string",
      "similarity_score": 0.95,
      "metadata": {}
    }
  ],
  "query_time": 0.123,
  "message": "Found X similar images"
}
```

## Error Handling

- Invalid image formats return 400 Bad Request
- Model loading errors return 500 Internal Server Error  
- All errors include descriptive messages

## Dependencies

Make sure to install:
```bash
pip install fastapi uvicorn python-multipart pillow
```
# Image Retrieval Frontend

Simple Streamlit web interface for the Image Retrieval System.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the API server is running:
```bash
cd ../source/api
python run_server.py
```

3. Start the Streamlit app:
```bash
python run_app.py
```

Or directly:
```bash
streamlit run app.py
```

## Usage

1. Open http://localhost:8501 in your browser
2. Upload an image file (JPG, PNG, JPEG)
3. Select model type and number of results
4. Click "Search Similar Images"
5. View results in the right panel

## Features

- **Upload & Search**: Upload images and find similar ones
- **Model Selection**: Choose between different feature extractors
- **Real-time API Status**: Check if backend is running
- **Configurable Results**: Adjust number of results returned
- **Health Monitoring**: Monitor API and model status

## API Endpoints Used

- `GET /health` - Check API status
- `POST /search/upload` - Search similar images
- `GET /models` - List available models
- `GET /vdb` - Database information
# Music Genre Classification API

A FastAPI-based machine learning API that predicts the genre of an uploaded audio file using a modified ResNet-50 deep learning model.

## Model

The model is a modified ResNet-50 trained for audio classification using mel spectrogram representations.

It classifies audio into 10 genres:

- blues
- classical
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock

The API converts the uploaded audio into a mel spectrogram and uses the trained ResNet-50 model to predict the genre.

### Model Files

- `model.pkl` — serialized model state dictionary used by the FastAPI application.
- `ResNet50_1.pth` — original PyTorch model checkpoint.

## API Endpoints

### `GET /health`

Checks whether the API and model are running.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`

Upload an audio file to receive a genre prediction.

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample.wav"
```

Illustrative response:

```json
{
  "prediction": "rock",
  "confidence": 0.91,
  "probabilities": {
    "blues": 0.01,
    "classical": 0.00,
    "country": 0.02,
    "disco": 0.01,
    "hiphop": 0.01,
    "jazz": 0.01,
    "metal": 0.01,
    "pop": 0.01,
    "reggae": 0.01,
    "rock": 0.91
  }
}
```

> **Note:** The response above is an illustrative example. Replace it with an actual response from the deployed API when documenting a real prediction.

## Running Locally

### 1. Create a virtual environment

Python 3.12 is recommended for this project.

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the API

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

## Docker

### Build the Docker image

```bash
docker build -t music-genre-api .
```

### Run the container

```bash
docker run -d -p 8000:8000 --name music-genre-container music-genre-api
```

The API will then be available at:

```text
http://localhost:8000
```

To view container logs:

```bash
docker logs music-genre-container
```

## Project Structure

```text
.
├── main.py
├── model.pkl
├── ResNet50_1.pth
├── model_conversion.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Technology Stack

- Python
- PyTorch
- Torchvision
- Librosa
- FastAPI
- Uvicorn
- Joblib
- Docker

## Deployment

The application is containerized using Docker and can be deployed to a cloud platform such as Render.

The live deployment URL will be added here after deployment.
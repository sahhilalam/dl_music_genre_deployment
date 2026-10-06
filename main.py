from fastapi import FastAPI
import torch
import torch.nn as nn
from torchvision import models
import joblib
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import librosa
import numpy as np
import io


GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]


# ------------------ MODEL ARCHITECTURE ------------------

class ResNet50Audio(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = models.resnet50(weights=None)

        # Accept 1-channel mel spectrograms
        self.model.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # 10 genre classes
        self.model.fc = nn.Linear(
            2048,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


# ------------------ LOAD MODEL ------------------

model = ResNet50Audio(num_classes=10)

state_dict = joblib.load("model.pkl")

model.load_state_dict(state_dict)

model.eval()


# ------------------ FASTAPI ------------------

app = FastAPI(
    title="Music Genre Classification API",
    description="ResNet-50 based audio genre classification API",
    version="1.0"
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }


@app.get("/predict", response_class=HTMLResponse)
async def predict_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <title>Music Genre Classifier</title>

        <style>
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                             Roboto, Helvetica, Arial, sans-serif;
                min-height: 100vh;
                background: linear-gradient(135deg, #0f172a, #1e293b);
                color: #f8fafc;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 30px;
            }

            .container {
                width: 100%;
                max-width: 650px;
            }

            .card {
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 24px;
                padding: 40px;
                box-shadow: 0 25px 60px rgba(0, 0, 0, 0.35);
            }

            .header {
                text-align: center;
                margin-bottom: 30px;
            }

            .icon {
                font-size: 48px;
                margin-bottom: 12px;
            }

            h1 {
                font-size: 32px;
                margin-bottom: 10px;
            }

            .subtitle {
                color: #cbd5e1;
                font-size: 15px;
                line-height: 1.6;
            }

            .upload-area {
                border: 2px dashed #64748b;
                border-radius: 18px;
                padding: 35px 20px;
                text-align: center;
                margin: 25px 0;
                transition: 0.2s;
            }

            .upload-area:hover {
                border-color: #38bdf8;
                background: rgba(56, 189, 248, 0.05);
            }

            input[type="file"] {
                display: none;
            }

            .file-label {
                display: inline-block;
                background: #0ea5e9;
                color: white;
                padding: 12px 24px;
                border-radius: 10px;
                cursor: pointer;
                font-weight: 600;
                transition: 0.2s;
            }

            .file-label:hover {
                background: #0284c7;
                transform: translateY(-1px);
            }

            #file-name {
                margin-top: 15px;
                color: #cbd5e1;
                font-size: 14px;
            }

            button {
                width: 100%;
                border: none;
                border-radius: 12px;
                padding: 14px;
                background: #22c55e;
                color: white;
                font-size: 16px;
                font-weight: 700;
                cursor: pointer;
                transition: 0.2s;
            }

            button:hover {
                background: #16a34a;
                transform: translateY(-1px);
            }

            button:disabled {
                background: #475569;
                cursor: not-allowed;
                transform: none;
            }

            #result {
                display: none;
                margin-top: 25px;
                padding: 25px;
                border-radius: 16px;
                background: rgba(15, 23, 42, 0.7);
            }

            .result-title {
                color: #94a3b8;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .prediction {
                font-size: 32px;
                font-weight: 800;
                margin: 8px 0;
                color: #38bdf8;
            }

            .confidence {
                color: #cbd5e1;
                margin-bottom: 20px;
            }

            .probability {
                margin-top: 10px;
            }

            .probability-header {
                display: flex;
                justify-content: space-between;
                font-size: 13px;
                margin-bottom: 5px;
            }

            .bar {
                height: 7px;
                background: #334155;
                border-radius: 10px;
                overflow: hidden;
            }

            .bar-fill {
                height: 100%;
                background: #38bdf8;
                border-radius: 10px;
            }

            .error {
                margin-top: 20px;
                padding: 15px;
                border-radius: 10px;
                background: rgba(239, 68, 68, 0.15);
                color: #fca5a5;
                display: none;
            }

            .footer {
                text-align: center;
                margin-top: 25px;
                color: #64748b;
                font-size: 12px;
            }
        </style>
    </head>

    <body>

        <div class="container">

            <div class="card">

                <div class="header">
                    <div class="icon">🎵</div>

                    <h1>Music Genre Classifier</h1>

                    <p class="subtitle">
                        Upload an audio file and let the ResNet-50 model
                        predict its musical genre.
                    </p>
                </div>

                <div class="upload-area">

                    <label for="audio-file" class="file-label">
                        Choose Audio File
                    </label>

                    <input
                        id="audio-file"
                        type="file"
                        accept="audio/*"
                    >

                    <p id="file-name">
                        No file selected
                    </p>

                </div>

                <button id="predict-btn" disabled>
                    Predict Genre
                </button>

                <div id="error" class="error"></div>

                <div id="result">

                    <div class="result-title">
                        Predicted Genre
                    </div>

                    <div id="prediction" class="prediction">
                        —
                    </div>

                    <div id="confidence" class="confidence">
                        Confidence: —
                    </div>

                    <div id="probabilities"></div>

                </div>

                <div class="footer">
                    Powered by ResNet-50 • FastAPI
                </div>

            </div>

        </div>

        <script>

            const fileInput = document.getElementById("audio-file");
            const fileName = document.getElementById("file-name");
            const predictButton = document.getElementById("predict-btn");
            const result = document.getElementById("result");
            const prediction = document.getElementById("prediction");
            const confidence = document.getElementById("confidence");
            const probabilities = document.getElementById("probabilities");
            const errorBox = document.getElementById("error");

            fileInput.addEventListener("change", () => {

                if (fileInput.files.length > 0) {

                    fileName.textContent =
                        fileInput.files[0].name;

                    predictButton.disabled = false;

                } else {

                    fileName.textContent =
                        "No file selected";

                    predictButton.disabled = true;
                }
            });


            predictButton.addEventListener("click", async () => {

                if (!fileInput.files.length) {
                    return;
                }

                predictButton.disabled = true;
                predictButton.textContent = "Analyzing...";

                result.style.display = "none";
                errorBox.style.display = "none";

                const formData = new FormData();

                formData.append(
                    "file",
                    fileInput.files[0]
                );

                try {

                    const response = await fetch(
                        "/predict",
                        {
                            method: "POST",
                            body: formData
                        }
                    );

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(
                            data.detail || "Prediction failed"
                        );
                    }

                    prediction.textContent =
                        data.prediction.toUpperCase();

                    confidence.textContent =
                        "Confidence: " +
                        (data.confidence * 100).toFixed(2) +
                        "%";

                    probabilities.innerHTML = "";

                    const sortedProbabilities =
                        Object.entries(data.probabilities)
                        .sort((a, b) => b[1] - a[1]);

                    sortedProbabilities.forEach(
                        ([genre, probability]) => {

                            const percentage =
                                probability * 100;

                            const div =
                                document.createElement("div");

                            div.className =
                                "probability";

                            div.innerHTML = `
                                <div class="probability-header">
                                    <span>${genre}</span>
                                    <span>${percentage.toFixed(2)}%</span>
                                </div>

                                <div class="bar">
                                    <div
                                        class="bar-fill"
                                        style="width: ${percentage}%"
                                    ></div>
                                </div>
                            `;

                            probabilities.appendChild(div);
                        }
                    );

                    result.style.display = "block";

                } catch (error) {

                    errorBox.textContent =
                        error.message;

                    errorBox.style.display =
                        "block";

                } finally {

                    predictButton.disabled = false;
                    predictButton.textContent =
                        "Predict Genre";
                }
            });

        </script>

    </body>
    </html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # Read uploaded audio
        audio_bytes = await file.read()

        # Load audio at 16 kHz
        y, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,
            mono=True
        )

        # 5-second window
        window_size = 16000 * 5

        # Make sure audio is long enough
        if len(y) < window_size:
            y = np.pad(
                y,
                (0, window_size - len(y))
            )

        probabilities = []

        # Process audio in 5-second chunks
        for start in range(0, len(y), window_size):

            chunk = y[start:start + window_size]

            if len(chunk) < window_size:
                chunk = np.pad(
                    chunk,
                    (0, window_size - len(chunk))
                )

            # Mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=chunk,
                sr=16000,
                n_fft=1024,
                hop_length=160,
                n_mels=128
            )

            # Convert to dB
            mel = librosa.power_to_db(
                mel,
                ref=np.max
            )

            # Convert to PyTorch tensor
            tensor = torch.tensor(
                mel,
                dtype=torch.float32
            )

            # Shape: (batch, channel, mel, time)
            tensor = tensor.unsqueeze(0).unsqueeze(0)

            # Model prediction
            with torch.no_grad():
                output = model(tensor)

                probs = torch.softmax(
                    output,
                    dim=1
                )

            probabilities.append(
                probs.squeeze(0).numpy()
            )

        # Average predictions across chunks
        avg_probs = np.mean(
            probabilities,
            axis=0
        )

        predicted_index = int(
            np.argmax(avg_probs)
        )

        prediction = GENRES[predicted_index]

        confidence = float(
            avg_probs[predicted_index]
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                genre: float(prob)
                for genre, prob in zip(
                    GENRES,
                    avg_probs
                )
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
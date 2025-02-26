# Emergency Analyzer

Emergency Analyzer is a FastAPI-based machine learning service that transcribes emergency calls using Whisper and classifies them using a fine-tuned BERT model. The system predicts the severity, type, and responsible entities based on the transcribed text.

## Features

- 📌 **Whisper-based transcription**: Converts audio files (M4A, MP3, WAV, etc.) into text.
- 🔍 **BERT-based classification**: Predicts the severity, type, and entities involved in the emergency.
- 🚀 **FastAPI integration**: Provides an API endpoint for seamless interaction.
- ⚡ **Optimized for performance**: Uses GPU (CUDA/MPS) when available.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/emergency-analyzer.git
   cd emergency-analyzer
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download and set up the Whisper and BERT models:
   ```sh
   python setup_models.py  # (Optional script to automate model downloads)
   ```

## Usage

### Start the API server
```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### 🔹 Transcription & Classification
- **POST `/predict_audio/`**
  - **Input**: Audio file (M4A, WAV, MP3, etc.)
  - **Output**:
    ```json
    {
      "transcription": "There is a fire in the building.",
      "gravita_predetta": "Alta",
      "tipologia_predetta": "Incendio",
      "enti_predetti": "Vigili del Fuoco"
    }
    ```

#### 🔹 Text-Based Classification
- **POST `/predict/`**
  - **Input**:
    ```json
    { "testo": "There is a robbery happening right now!" }
    ```
  - **Output**:
    ```json
    {
      "gravita_predetta": "Alta",
      "tipologia_predetta": "Rapina",
      "enti_predetti": "Polizia"
    }
    ```

## Model Details

- **Transcription**: Uses OpenAI's `whisper-medium` model.
- **Classification**: Fine-tuned `dbmdz/bert-base-italian-uncased` model with three classifiers:
  - **Severity (`gravita`)**: Low, Medium, High
  - **Type (`tipologia`)**: Road accident, Fire, Theft, Illness, Assault, Robbery, Flood
  - **Entities (`enti`)**: Mapped dynamically from training data

## Troubleshooting

- If you encounter `torch not found` errors, ensure PyTorch is installed with the correct CUDA/MPS support:
  ```sh
  pip install torch torchvision torchaudio
  ```
- If transcription fails, check that `ffmpeg` is installed:
  ```sh
  sudo apt install ffmpeg  # Linux
  brew install ffmpeg       # macOS
  ```

## Future Improvements

- ✅ Expand entity recognition to include more agencies
- ✅ Improve dataset for better classification accuracy
- ✅ Optimize audio preprocessing for low-quality recordings

## License

This project is licensed under the MIT License.

## Contributors

- *Eustachio Fontana* - [GitHub Profile](https://github.com/UccioFontana)


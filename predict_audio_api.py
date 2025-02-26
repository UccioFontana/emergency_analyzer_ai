from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import BertTokenizer, BertModel, WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import tempfile
import os
from pydub import AudioSegment

print(torchaudio.list_audio_backends())

torchaudio.set_audio_backend("soundfile")

app = FastAPI()

# ✅ Modello Whisper per la trascrizione
WHISPER_MODEL_NAME = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Dispositivo per BERT
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_SAVE_PATH = "./saved_model"

# ✅ Tokenizer BERT e caricamento modello
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")

class BertMultiTask(torch.nn.Module):
    def __init__(self, base_model_name):
        super(BertMultiTask, self).__init__()
        self.bert = BertModel.from_pretrained(base_model_name)
        self.gravita_classifier = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.tipologia_classifier = torch.nn.Linear(self.bert.config.hidden_size, 7)
        self.enti_classifier = torch.nn.Linear(self.bert.config.hidden_size, 8)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output.contiguous()
        return {
            "gravita": self.gravita_classifier(pooled_output),
            "tipologia": self.tipologia_classifier(pooled_output),
            "enti": self.enti_classifier(pooled_output)
        }

# ✅ Carica modello BERT fine-tunato
model = BertMultiTask("dbmdz/bert-base-italian-uncased").to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "model.pt"), map_location=device))

def transcribe_audio(audio_file: UploadFile) -> str:
    """Trascrive audio multiformato con Whisper, convertendo M4A in WAV se necessario."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_audio:
            temp_audio.write(audio_file.file.read())
            temp_audio_path = temp_audio.name

        file_ext = os.path.splitext(temp_audio_path)[1].lower()
        
        # ✅ Se il formato è M4A, convertirlo in WAV
        if file_ext == ".m4a":
            audio = AudioSegment.from_file(temp_audio_path, format="m4a")
            wav_temp_path = temp_audio_path.replace(".m4a", ".wav")
            audio.export(wav_temp_path, format="wav")
            os.remove(temp_audio_path)  # 🔄 Rimuove il file M4A originale
            temp_audio_path = wav_temp_path

        # 🔄 Carica e converte l’audio a 16 kHz mono
        waveform, sample_rate = torchaudio.load(temp_audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        if waveform.shape[0] > 1:  # Se stereo, converti a mono
            waveform = waveform.mean(dim=0, keepdim=True)

        input_features = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_features.to(whisper_model.device)
        predicted_ids = whisper_model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore nella trascrizione: {e}")
    finally:
        os.remove(temp_audio_path)  # 🧹 Pulisce il file temporaneo

    return transcription.strip()

@app.post("/predict_audio/")
async def predict_audio(file: UploadFile = File(...)):
    """Riceve un file audio, lo trascrive e predice le etichette."""
    supported_formats = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in supported_formats:
        raise HTTPException(status_code=400, detail=f"Formato non supportato. Supportati: {supported_formats}")

    transcription = transcribe_audio(file)
    inputs = tokenizer(transcription, return_tensors="pt", padding=True, truncation=True, max_length=32)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        gravita_pred = torch.argmax(outputs["gravita"], dim=1).item()
        tipologia_pred = torch.argmax(outputs["tipologia"], dim=1).item()
        enti_pred = torch.argmax(outputs["enti"], dim=1).item()

    gravita_reverse = {0: "Bassa", 1: "Media", 2: "Alta"}
    tipologia_reverse = {0: "Incidente stradale", 1: "Incendio", 2: "Furto", 3: "Malore", 4: "Aggressione", 5: "Rapina", 6: "Allagamento"}
    
    import json
    with open("enti_labels.json", "r") as f:
        enti_labels = json.load(f)


    enti_reverse = {v: k for k, v in enti_labels.items()}  # Invertito per il mapping corretto

    return {
        "trascrizione": transcription,
        "gravita_predetta": gravita_reverse.get(gravita_pred, "Sconosciuto"),
        "tipologia_predetta": tipologia_reverse.get(tipologia_pred, "Sconosciuto"),
        "enti_predetti": enti_reverse.get(enti_pred, "Sconosciuto")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

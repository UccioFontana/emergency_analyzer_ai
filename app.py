from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import torch
import pandas as pd
import os
import gc

app = FastAPI()

class InputText(BaseModel):
    testo: str

# 🔥 Disabilita SafeTensors per evitare problemi di contiguità
os.environ["SAFETENSORS_FAST"] = "0"

# Forza l'uso di MPS su macOS, altrimenti usa CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando il dispositivo: {device}")

MODEL_SAVE_PATH = "./saved_model"

df = pd.read_csv("mnt/data/chiamate_emergenza.csv")

gravita_labels = {"Bassa": 0, "Media": 1, "Alta": 2}
tipologia_labels = {"Incidente stradale": 0, "Incendio": 1, "Furto": 2, "Malore": 3, "Aggressione": 4, "Rapina": 5, "Allagamento": 6}
enti_unici = set(df["enti"].unique())

enti_labels = {ente: i for i, ente in enumerate(sorted(enti_unici))}

# ✅ Salva enti_labels in un file JSON
import json
with open("enti_labels.json", "w") as f:
    json.dump(enti_labels, f)

df["gravita_label"] = df["gravita"].map(gravita_labels).astype(int)
df["tipologia_label"] = df["tipologia"].map(tipologia_labels).astype(int)
df["enti_label"] = df["enti"].map(enti_labels).astype(int)

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")

def tokenize_function(example):
    return tokenizer(example["testo"], padding="max_length", truncation=True, max_length=32)

dataset = Dataset.from_pandas(df[["testo", "gravita_label", "tipologia_label", "enti_label"]])
tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def format_labels(examples):
    return {
        "input_ids": examples["input_ids"],
        "attention_mask": examples["attention_mask"],
        "gravita_label": examples["gravita_label"],
        "tipologia_label": examples["tipologia_label"],
        "enti_label": examples["enti_label"]
    }

tokenized_dataset = tokenized_dataset.map(format_labels, batched=True)

class BertMultiTask(torch.nn.Module):
    def __init__(self, base_model_name):
        super(BertMultiTask, self).__init__()
        self.bert = BertModel.from_pretrained(base_model_name)
        self.gravita_classifier = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.tipologia_classifier = torch.nn.Linear(self.bert.config.hidden_size, 7)
        self.enti_classifier = torch.nn.Linear(self.bert.config.hidden_size, len(enti_labels))  # Dinamico

    def forward(self, input_ids, attention_mask=None, gravita_label=None, tipologia_label=None, enti_label=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output.contiguous()  # 🔥 Rende il tensore contiguo

        gravita_logits = self.gravita_classifier(pooled_output).contiguous()
        tipologia_logits = self.tipologia_classifier(pooled_output).contiguous()
        enti_logits = self.enti_classifier(pooled_output).contiguous()

        loss = None
        if gravita_label is not None and tipologia_label is not None and enti_label is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = (
                loss_fct(gravita_logits, gravita_label) +
                loss_fct(tipologia_logits, tipologia_label) +
                loss_fct(enti_logits, enti_label)
            )

        return {"loss": loss, "gravita": gravita_logits, "tipologia": tipologia_logits, "enti": enti_logits}

# Caricare o addestrare il modello
if os.path.exists(os.path.join(MODEL_SAVE_PATH, "model.pt")):
    model = BertMultiTask("dbmdz/bert-base-italian-uncased").to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "model.pt"), map_location=device))
else:
    model = BertMultiTask("dbmdz/bert-base-italian-uncased").to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        save_strategy="no",
        logging_dir="./logs",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    trainer.train()
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # 🔥 Funzione per rendere i tensori contigui prima del salvataggio
    def save_model_fix(model, path):
        state_dict = model.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].contiguous()  # Forza la contiguità del tensore
        torch.save(state_dict, path)

    save_model_fix(model, os.path.join(MODEL_SAVE_PATH, "model.pt"))

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

@app.post("/predict")
def predict(input_text: InputText):
    inputs = tokenizer(input_text.testo, return_tensors="pt", padding=True, truncation=True, max_length=32)
    inputs = {key: value.to(device) for key, value in inputs.items() if key != "token_type_ids"}  # Rimuove token_type_ids

    print(f"Dispositivo input: {inputs['input_ids'].device}")
    print(f"Dispositivo modello: {next(model.parameters()).device}")

    model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs)
        gravita_pred = torch.argmax(outputs["gravita"], dim=1).item()
        tipologia_pred = torch.argmax(outputs["tipologia"], dim=1).item()
        enti_pred = torch.argmax(outputs["enti"], dim=1).item()

    gravita_reverse = {v: k for k, v in gravita_labels.items()}
    tipologia_reverse = {v: k for k, v in tipologia_labels.items()}
    enti_reverse = {v: k for k, v in enti_labels.items()}

    return {
        "gravita_predetta": gravita_reverse.get(gravita_pred, "Sconosciuto"),
        "tipologia_predetta": tipologia_reverse.get(tipologia_pred, "Sconosciuto"),
        "enti_predetti": enti_reverse.get(enti_pred, "Sconosciuto")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

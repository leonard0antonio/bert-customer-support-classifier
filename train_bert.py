# Fine-tuning BERT para Classificação de Atendimento ao Cliente
# Classifica mensagens em "venda" ou "suporte"

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

# Configurações
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"  # BERT em português
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./bert-customer-support-model"

# Mapeamento de labels
LABEL_MAP = {"venda": 0, "suporte": 1}
ID2LABEL = {0: "venda", 1: "suporte"}


class CustomerSupportDataset(Dataset):
    """Dataset para classificação de mensagens de atendimento"""
    
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.texts.append(data['prompt'])
                self.labels.append(LABEL_MAP[data['completion']])
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Calcula métricas de avaliação"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
    }


def train_model():
    """Treina o modelo BERT para classificação"""
    
    print("Carregando tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    print("Carregando datasets...")
    train_dataset = CustomerSupportDataset('treino.jsonl', tokenizer, MAX_LENGTH)
    test_dataset = CustomerSupportDataset('teste.jsonl', tokenizer, MAX_LENGTH)
    
    print(f"Tamanho do treino: {len(train_dataset)}")
    print(f"Tamanho do teste: {len(test_dataset)}")
    
    print("Carregando modelo...")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL_MAP
    )
    
    # Configurações de treinamento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        learning_rate=LEARNING_RATE,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Iniciando treinamento...")
    trainer.train()
    
    print("Salvando modelo...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Avaliando no conjunto de teste...")
    results = trainer.evaluate()
    print(f"Resultados: {results}")
    
    return trainer, tokenizer


def predict(text, model, tokenizer):
    """Faz predição para um texto"""
    model.eval()
    
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    return ID2LABEL[predicted_class], confidence


def test_predictions(trainer, tokenizer):
    """Testa o modelo com exemplos"""
    test_cases = [
        "Olá, gostaria de comprar o novo smartphone",
        "Meu produto veio com defeito",
        "Quero fazer um pedido da geladeira",
        "Como configurar a smart TV?",
        "Tenho interesse no fogão 5 bocas",
        "Preciso de ajuda com a garantia",
        "Gostaria de adquirir a air fryer",
        "O liquidificador não liga",
        "Quero comprar o notebook gamer",
        "Como faço para devolver um produto?"
    ]
    
    print("\n=== Testes de Predição ===")
    for text in test_cases:
        label, confidence = predict(text, trainer.model, tokenizer)
        print(f"Texto: {text}")
        print(f"Classificação: {label} (confiança: {confidence:.4f})")
        print("-" * 50)


if __name__ == "__main__":
    # Verifica se CUDA está disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    
    # Treina o modelo
    trainer, tokenizer = train_model()
    
    # Testa predições
    test_predictions(trainer, tokenizer)
    
    print("\nTreinamento concluído! Modelo salvo em:", OUTPUT_DIR)
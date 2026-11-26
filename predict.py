#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de inferência para o modelo BERT de classificação de atendimento ao cliente.
Uso: python predict.py "mensagem do cliente"
     python predict.py  (modo interativo)
"""

import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "./bert-customer-support-model"
MAX_LENGTH = 128
ID2LABEL = {0: "venda", 1: "suporte"}


def load_model():
    """Carrega modelo e tokenizer"""
    print(f"Carregando modelo de {MODEL_PATH}...")
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        print("Modelo carregado com sucesso!")
        return model, tokenizer
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        print("Certifique-se de que o modelo foi treinado e salvo em", MODEL_PATH)
        sys.exit(1)


def predict(text, model, tokenizer):
    """Faz predição para um texto"""
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**encoding)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        all_probs = probabilities[0].tolist()
    
    return {
        'label': ID2LABEL[predicted_class],
        'confidence': confidence,
        'probabilities': {
            'venda': all_probs[0],
            'suporte': all_probs[1]
        }
    }


def interactive_mode(model, tokenizer):
    """Modo interativo para testar múltiplas mensagens"""
    print("\n=== Modo Interativo ===")
    print("Digite mensagens para classificar (Ctrl+C para sair)")
    print("-" * 50)
    
    while True:
        try:
            text = input("\nMensagem: ").strip()
            if not text:
                continue
            
            result = predict(text, model, tokenizer)
            print(f"Classificação: {result['label'].upper()}")
            print(f"Confiança: {result['confidence']:.4f}")
            print(f"Probabilidades: Venda={result['probabilities']['venda']:.4f}, Suporte={result['probabilities']['suporte']:.4f}")
            
        except KeyboardInterrupt:
            print("\n\nSaindo...")
            break
        except EOFError:
            break


def main():
    if len(sys.argv) > 1:
        # Modo linha de comando
        text = " ".join(sys.argv[1:])
        model, tokenizer = load_model()
        result = predict(text, model, tokenizer)
        print(f"Texto: {text}")
        print(f"Classificação: {result['label'].upper()}")
        print(f"Confiança: {result['confidence']:.4f}")
        print(f"Probabilidades: Venda={result['probabilities']['venda']:.4f}, Suporte={result['probabilities']['suporte']:.4f}")
    else:
        # Modo interativo
        model, tokenizer = load_model()
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
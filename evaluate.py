#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de avaliação completa do modelo no conjunto de teste.
Gera relatório de classificação e matriz de confusão.
"""

import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from train_bert import CustomerSupportDataset, LABEL_MAP, ID2LABEL, MAX_LENGTH

MODEL_PATH = "./bert-customer-support-model"
BATCH_SIZE = 16


def evaluate_model():
    """Avalia o modelo no conjunto de teste"""
    print("Carregando modelo e tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    
    print("Carregando dataset de teste...")
    test_dataset = CustomerSupportDataset('teste.jsonl', tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    print("Executando predições...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Métricas
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print("\n" + "="*60)
    print("RELATÓRIO DE AVALIAÇÃO")
    print("="*60)
    print(f"\nAcurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nRelatório de Classificação:")
    print(classification_report(
        all_labels, 
        all_predictions, 
        target_names=['venda', 'suporte'],
        digits=4
    ))
    
    print("\nMatriz de Confusão:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"                 Predito")
    print(f"              venda  suporte")
    print(f"Real venda     {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"     suporte   {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Análise por classe
    print("\nAnálise por Classe:")
    for i, label in enumerate(['venda', 'suporte']):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{label.upper()}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Support:   {cm[i, :].sum()}")
    
    # Exemplos de erros
    print("\n" + "="*60)
    print("EXEMPLOS DE ERROS (se houver)")
    print("="*60)
    
    with open('teste.jsonl', 'r', encoding='utf-8') as f:
        test_texts = [json.loads(line)['prompt'] for line in f]
    
    errors = [(i, test_texts[i], ID2LABEL[all_labels[i]], ID2LABEL[all_predictions[i]]) 
              for i in range(len(all_labels)) if all_labels[i] != all_predictions[i]]
    
    if errors:
        for idx, text, true_label, pred_label in errors:
            print(f"\nTexto: {text}")
            print(f"  Verdadeiro: {true_label} | Predito: {pred_label}")
    else:
        print("\nNenhum erro! Modelo acertou todos os exemplos de teste.")
    
    return accuracy


if __name__ == "__main__":
    try:
        evaluate_model()
    except FileNotFoundError:
        print(f"Modelo não encontrado em {MODEL_PATH}")
        print("Execute primeiro: python train_bert.py")
    except Exception as e:
        print(f"Erro durante avaliação: {e}")
        import traceback
        traceback.print_exc()
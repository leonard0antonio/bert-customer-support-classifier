# Fine-tuning BERT para Classificação de Atendimento ao Cliente
# Classifica mensagens em "venda" ou "suporte"

## Visão Geral

Este projeto implementa fine-tuning de um modelo BERT em português (neuralmind/bert-base-portuguese-cased) 
para classificar mensagens de clientes de uma rede varejista em duas categorias:
- **venda**: Intenção de compra de produto
- **suporte**: Dúvidas ou problemas com produto

## Estrutura dos Arquivos

```
bert-customer-support-classifier/
├── treino.jsonl          # Dataset de treino (50 exemplos)
├── teste.jsonl           # Dataset de teste (10 exemplos)
├── train_bert.py         # Script principal de treinamento
├── requirements.txt      # Dependências
├── predict.py            # Script para inferência
└── README.md             # Este arquivo
```

## Instalação

```bash
# Criar ambiente virtual (recomendado)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependências
pip install -r requirements.txt
```

## Dependências (requirements.txt)

```
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
tqdm>=4.65.0
accelerate>=0.20.0
```

## Treinamento

```bash
python train_bert.py
```

O script irá:
1. Carregar o tokenizer e modelo BERT base em português
2. Preparar os datasets de treino e teste
3. Treinar por 4 epochs com batch size 16
4. Salvar o modelo em `./bert-customer-support-model/`
5. Exibir métricas de avaliação
6. Testar com exemplos de predição

## Inferência (predict.py)

```bash
python predict.py "Sua mensagem aqui"
```

Ou use interativamente:
```bash
python predict.py
```

## Formato dos Dados (JSONL)

Cada linha é um JSON com:
- `prompt`: Texto da mensagem do cliente
- `completion`: Label ("venda" ou "suporte")

Exemplo:
```json
{"prompt": "Olá, gostaria de fazer a aquisição do novo produto", "completion": "venda"}
{"prompt": "tudo bom, queria verificar como funciona a TV Smart x0912", "completion": "suporte"}
```

## Modelo Utilizado

- **Base**: `neuralmind/bert-base-portuguese-cased` (BERT treinado em português)
- **Task**: Sequence Classification (2 classes)
- **Max Length**: 128 tokens
- **Learning Rate**: 2e-5
- **Epochs**: 4
- **Batch Size**: 16

## Métricas Esperadas

Com o dataset fornecido (50 treino + 10 teste), espera-se:
- Accuracy > 90% no conjunto de teste
- Boa generalização para mensagens similares

## Personalização

Para usar seus próprios dados:
1. Substitua `treino.jsonl` e `teste.jsonl` pelos seus arquivos
2. Mantenha o mesmo formato JSONL
3. Ajuste hiperparâmetros em `train_bert.py` se necessário:
   - `EPOCHS`: Mais epochs para datasets maiores
   - `BATCH_SIZE`: Reduza se tiver problemas de memória
   - `LEARNING_RATE`: 2e-5 a 5e-5 típico para BERT
   - `MAX_LENGTH`: Aumente para textos mais longos

## Salvando e Carregando o Modelo

O modelo é salvo automaticamente em `./bert-customer-support-model/`.

Para carregar em outro script:
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("./bert-customer-support-model")
tokenizer = BertTokenizer.from_pretrained("./bert-customer-support-model")

# Predição
inputs = tokenizer("sua mensagem", return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=-1).item()
    labels = ["venda", "suporte"]
    print(labels[pred])
```

## Requisitos de Hardware

- **GPU**: Recomendada (CUDA) para treinamento mais rápido
- **CPU**: Funciona, mas mais lento
- **RAM**: Mínimo 8GB, recomendado 16GB+
- **VRAM**: 4GB+ para batch_size=16

## Solução de Problemas

### Erro de memória (OOM)
```python
# Reduza batch size
BATCH_SIZE = 8  # ou 4
```

### Erro de conexão HuggingFace
```bash
# Configure proxy se necessário
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

### Python não encontrado
- Verifique se Python está no PATH
- Use `python -m pip` ou `py -m pip`
- Reinstale Python se necessário

## Licença

Projeto educacional para fins didáticos. Dados sintéticos.
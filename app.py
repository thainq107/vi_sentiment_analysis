import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")
idx2label = {0: 'negative', 1:'positive'}

def load_model(model_path, vocab_size=10000, embedding_dim=100, num_classes=2):
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        kernel_sizes=[3, 4, 5],
        num_filters=100,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    return model
  
model = load_model('text_cnn_model.pt')
vocabulary = torch.load('vocabulary.pth')

def inference(sentence, vocabulary, model):
    encoded_sentence = vocabulary(tokenizer(sentence))
    encoded_sentence = torch.tensor(encoded_sentence)
    encoded_sentence = torch.unsqueeze(encoded_sentence, 1)

    with torch.no_grad():
        predictions = model(encoded_sentence)
    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)
    return round(p_max.item(), 2)*100, yhat.item()

def main():
  st.title('Sentiment Analysis')
  st.title('Model: Text CNN. Dataset: NTC-SCV')
  text_input = st.text_input("Sentence: ", "Đồ ăn ở quán này quá tệ luôn!")
  p, idx = inference(text_input, vocabulary, model)
  label = idx2label[idx]
  st.success(f'Sentiment: {label} with {p:.2f} % probability.') 

if __name__ == '__main__':
     main() 

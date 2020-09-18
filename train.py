import time
import numpy as np
import pandas as pd

import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertPreTrainedModel, BertModel
from transformers import get_linear_schedule_with_warmup

import config
from utilitors import preprocessing

def attention_masks(encoded_sentences):
    # attention masks, 0 for padding, 1 for actual token
    attention_masks = []
    for sent in encoded_sentences:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks

device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print(f'Found GPU at: {device_name}')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU in use:', torch.cuda.get_device_name(0))
else:
    print('using the CPU')
    device = torch.device("cpu")


def main():

    device_name = tf.test.gpu_device_name()
    if device_name == '/device:GPU:0':
        print(f'Found GPU at: {device_name}')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('GPU in use:', torch.cuda.get_device_name(0))
    else:
        print('using the CPU')
        device = torch.device("cpu")

    # load the datasets
    df = pd.read_csv("./english.train.10000", delimiter='\t', header=None,
                     names=['label', 'sentence'])
    df_test = pd.read_csv("./english.dev", delimiter='\t', header=None, names=['label', 'sentence'])

    train_encoded_sentences, train_labels = preprocessing(df)
    train_attention_masks = attention_masks(train_encoded_sentences)

    test_encoded_sentences, test_labels = preprocessing(df_test)
    test_attention_masks = attention_masks(test_encoded_sentences)


if __name__ == "__main__":
    main()

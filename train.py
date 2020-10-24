import time
import numpy as np
import pandas as pd
import random

import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertPreTrainedModel, BertModel
from transformers import get_linear_schedule_with_warmup

import config
from utilitors import preprocessing, create_attention_masks, run_train, loss_plot, save_trained_model, run_evaluation


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
    train_attention_masks = create_attention_masks(train_encoded_sentences)

    test_encoded_sentences, test_labels = preprocessing(df_test)
    test_attention_masks = create_attention_masks(test_encoded_sentences)

    train_inputs = torch.tensor(train_encoded_sentences)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_attention_masks)

    validation_inputs = torch.tensor(test_encoded_sentences)
    validation_labels = torch.tensor(test_labels)
    validation_masks = torch.tensor(test_attention_masks)

    # data loader for training
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)

    # data loader for validation
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=config.batch_size)

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=4,
        output_attentions=False,
        output_hidden_states=False,
    )

    model.cuda()

    optimizer = AdamW(model.parameters(),
                      lr=3e-5,
                      eps=1e-8,
                      weight_decay=0.01
                      )

    epochs = 3
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # 10% * datasetSize/batchSize
                                                num_training_steps=total_steps)

    losses, model = run_train(model, train_dataloader, validation_dataloader, device, epochs, optimizer)
    loss_plot(losses)
    save_trained_model(model)

    print("Evaluating on english \n")
    df_test = pd.read_csv("./english.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_evaluation(df_test)

    print("Evaluating on spanish \n")
    df_test = pd.read_csv("./spanish.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_evaluation(df_test)

    print("Evaluating on french \n")
    df_test = pd.read_csv("./french.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_evaluation(df_test)

    print("Evaluating on italian \n")
    df_test = pd.read_csv("./italian.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_evaluation(df_test)

    print("Evaluating on japanese \n")
    df_test = pd.read_csv("./japanese.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_evaluation(df_test)

    print("Evaluating on russian \n")
    df_test = pd.read_csv("./russian.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_evaluation(df_test)

    print("Evaluating on german \n")
    df_test = pd.read_csv("./german.test", delimiter='\t', header=None, names=['label', 'sentence'])
    run_evaluation(df_test)


if __name__ == "__main__":
    main()

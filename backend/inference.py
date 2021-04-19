import config
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer


def preprocessing(df):
    sentences = df.sentence.values
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

    encoded_sentences = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
            truncation=True,
            max_length=config.MAX_LEN
        )

        encoded_sentences.append(encoded_sent)
    encoded_sentences = pad_sequences(encoded_sentences, maxlen=config.MAX_LEN, dtype="long",
                                      value=0, truncating="post", padding="post")
    return encoded_sentences


def attention_masks(encoded_sentences):
    # attention masks, 0 for padding, 1 for actual token
    attention_masks = []
    for sent in encoded_sentences:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks


def inference(doc, model):
    device = torch.device("cpu")
    doc_df = pd.DataFrame({"sentence": [doc]})

    encoded_doc = preprocessing(doc_df)

    test_attention_masks = attention_masks(encoded_doc)

    test_inputs = torch.tensor(encoded_doc)
    test_masks = torch.tensor(test_attention_masks)

    test_data = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=128)

    model.eval()
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        data, masks = batch
        with torch.no_grad():
            out = model(data,
                        token_type_ids=None,
                        attention_mask=masks)
        logits = out[0]
        logits = logits.detach().cpu().numpy()
        predicted_class = np.argmax(logits, axis=1).flatten()[0]

    return config.LABELS[predicted_class]

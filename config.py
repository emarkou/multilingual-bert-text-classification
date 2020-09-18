MAX_LEN = 128 # max sequences length
batch_size = 32

# extra preprocessing steps
# prepend CLS and append SEP, truncate, pad

labels_encoding = {
    "CCAT": 0,
    "ECAT": 1,
    "GCAT": 2,
    "MCAT": 3
}

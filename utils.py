import wget
import os
import torch

def load_dataset(filename="input.txt", path="./data"):
    """
    loads the shakespeare dataset from a specified path and filename.

    if the filename is not available downloads it from github
    """
    if filename not in os.listdir(path):
        dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

        f = wget.download(dataset_url, out=path)

    with open(os.path.join(path,filename),"r") as fn:
        data = fn.read()

    return data

def encode(input_string, vocabulary):
    stoi = {char:i for i,char in enumerate(vocabulary)}
    return [stoi[char] for char in input_string]

def decode(token_list, vocabulary):
    itos = {i:char for i,char in enumerate(vocabulary)}
    return "".join([itos[token] for token in token_list])

def get_batch(input_tensor, context_length, batch_size):

    idx = torch.randint(len(input_tensor)-context_length, (batch_size,))

    x = torch.stack([input_tensor[i  :i+context_length  ] for i in idx])
    y = torch.stack([input_tensor[i+1:i+context_length+1] for i in idx])

    return x,y


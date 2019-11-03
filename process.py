import numpy as np
import unidecode
import pdb
import torch.utils.data

from utils import *
from Models import *


"""
Some parts of the pre processing were extracted from https://www.tensorflow.org/tutorials/text/text_generation
"""

class Int2OneHot():
    def __init__(self, number_of_classes, device="cpu"):
        self.nc = number_of_classes
        self.matrix = torch.eye(number_of_classes, device=device)

    def create(self, x):
        return self.matrix[x]


seed = 1111
seed_everything(seed)
dpi = 400

parser = argparse.ArgumentParser(description="pineraBot")
parser.add_argument('--bs', type=int, default=30, help='input batch size for training (default: 128)')
parser.add_argument('--e', type=int, default=2, help='number of epochs to train (default: 10)')
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default 2e-4)")
parser.add_argument("--pre", action="store_true", help="train pre trained model (default False)")
parser.add_argument('--hs', type=int, default=2, help="hidden size (default: 2)")
parser.add_argument('--nl', type=int, default=1, help="number of layers (default: 1)")
args = parser.parse_args()

path_to_file = "shakespeare.txt"
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_len = 100
number_of_examples = len(text) // (seq_len + 1)

# Remove last part

text_as_int = text_as_int[:int(number_of_examples * (seq_len + 1))]

# Split text as sequences of len seq_len

text_as_int = text_as_int.reshape((number_of_examples, seq_len + 1))

# Create input and output sequences

x = text_as_int[:, :seq_len]
y = text_as_int[:, 1:]

# Create dataset and dataloader

dataset = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True)

pdb.set_trace()
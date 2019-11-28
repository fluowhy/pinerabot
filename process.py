import numpy as np
import unidecode
import pdb
import torch.utils.data
import argparse
from tqdm import tqdm
from torch.distributions.categorical import Categorical

from utils import *
from models import *


"""
Some parts of the pre processing were extracted from https://www.tensorflow.org/tutorials/text/text_generation
"""

seed = 1111
seed_everything(seed)
dpi = 400

parser = argparse.ArgumentParser(description="pineraBot")
parser.add_argument('--bs', type=int, default=30, help='input batch size for training (default: 128)')
parser.add_argument('--e', type=int, default=2, help='number of epochs to train (default: 2)')
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default 2e-4)")
parser.add_argument("--pre", action="store_true", help="train pre trained model (default False)")
parser.add_argument('--nh', type=int, default=2, help="hidden size (default: 2)")
parser.add_argument('--nlayers', type=int, default=1, help="number of layers (default: 1)")
parser.add_argument("--do", type=float, default=0., help="dropout (default 0.)")
parser.add_argument("--wd", type=float, default=0., help="weight decay (default 0.)")
args = parser.parse_args()

path_to_file = "pinera.txt"
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

# Split train/val sets

val_ratio = 0.1
n_val = int(len(x) * val_ratio)
index = np.arange(len(x))
np.random.shuffle(index)

x_val = x[:n_val]
x_train = x[n_val:]

y_val = y[:n_val]
y_train = y[n_val:]

# Create dataset and dataloader

train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.long, device="cpu"), torch.tensor(y_train, dtype=torch.long, device="cpu"))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(torch.tensor(x_val, dtype=torch.long, device="cpu"), torch.tensor(y_val, dtype=torch.long, device="cpu"))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=True)


class TextModel():
    def __init__(self, nin, nh, nlayers, nout, do, device, pre):
        self.model = LSTM(nin, nh, nlayers, nout, do)
        self.model_name = "lstm"
        self.device = device
        self.model.to(device)
        self.best_loss = np.inf
        self.load_model() if pre else 0
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none").to(device)
        print("model params {}".format(count_parameters(self.model)))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.to_device = ToDevice(device)
        self.label2onehot = Label2OneHot(nout)
        self.print_template = "Epoch {:03d} | Train loss {:.3f} | Test loss {:.3f} | Val loss {:.3f}"

    def train_model(self, dataloader):
        self.model.train()
        train_loss = 0
        for idx, (x, y) in enumerate(tqdm(dataloader)):
            self.optimizer.zero_grad()
            x, y = self.to_device([x, y])
            x = self.label2onehot(x)
            y_pred = self.model(x).transpose(1, 2)
            loss = self.cross_entropy(y_pred, y).sum(-1).mean()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= (idx + 1)
        return train_loss

    def eval_model(self, dataloader):
        self.model.eval()
        eval_loss = 0
        for idx, (x, y) in enumerate(tqdm(dataloader)):
            x, y = self.to_device([x, y])
            x = self.label2onehot(x)
            y_pred = self.model(x).transpose(1, 2)
            loss = self.cross_entropy(y_pred, y).sum(-1).mean()
            eval_loss += loss.item()
        eval_loss /= (idx + 1)
        return eval_loss

    def fit(self, train_loader, val_loader, lr, wd):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        losses = []
        for epoch in range(args.e):
            train_loss = self.train_model(train_loader)
            val_loss = self.eval_model(val_loader)
            losses.append((train_loss, val_loss))
            print(self.print_template.format(epoch, train_loss, 0, val_loss))
            if val_loss < self.best_loss:
                print("Saving")
                torch.save(self.model.state_dict(), "models/{}.pth".format(self.model_name))
                self.best_loss = train_loss
            np.save("files/losses", losses)
        return

    def generate(self, start_string, n_chars):
        print("Generating sentence")
        self.load_model()
        self.model.eval()
        start_integers = np.array([char2idx[c] for c in start_string])[np.newaxis]
        start_integers = torch.tensor(start_integers, dtype=torch.long, device=self.device)        
        with torch.no_grad():
            for i in tqdm(range(n_chars)):
                x = self.label2onehot(start_integers)
                y = self.model(x)[:, -1]
                y = self.softmax(y).squeeze()
                y = Categorical(y).sample().reshape((1, 1))
                start_integers = torch.cat((start_integers, y), dim=-1)
        sentence = idx2char[start_integers.cpu().squeeze().numpy()]
        sentence = "".join(sentence)        
        return sentence

    def save_model(self):
        torch.save(self.model.state_dict(), "models/{}.pth".format(self.model_name))
        return

    def load_model(self):
        self.model.load_state_dict(torch.load("models/{}.pth".format(self.model_name), map_location=self.device))
        return

text_model = TextModel(
    nin=len(vocab),
    nh=args.nh,
    nlayers=args.nlayers,
    nout=len(vocab),
    do=args.do,
    device=args.d,
    pre=args.pre
    )

text_model.fit(train_dataloader, val_dataloader, lr=args.lr, wd=args.wd,)

sentence = text_model.generate(start_string=u"Muy ", n_chars=200)
print(sentence)

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import argparse
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.utils.data
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Models import *


def myFunc(e):
  return len(e)

#seed = 1111
#seed_everything(seed)
parser = argparse.ArgumentParser(description="pineraBot")
parser.add_argument('--batch_size', type=int, default=100, metavar='B', help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs to train (default: 10)')
parser.add_argument("--d", tyep=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--lr", type=float, default=1e-3, metavar="L", help="learning rate")
parser.add_argument("--pre", action="store_true", help="train pre trained model (default False)")
parser.add_argument("--debug", action="store_true", help="enables debug (default False")
args = parser.parse_args()

device = args.d
print(device)

labels = np.load("labels.npy")
eos = int(np.nonzero(labels=="<EOS>")[0][0])
pad = int(np.nonzero(labels=="<PAD>")[0][0])
num = int(np.nonzero(labels=="<NUM>")[0][0])

nlabels = len(labels)
sentences = np.load("num_sentences.npy")
x_train = []
y_train = []
lengths = []
for i, sen in enumerate(sentences):
	lengths.append(len(sen))
	x_train.append(torch.tensor(sen[:-1], dtype=torch.long).to(device))
	y_train.append(torch.tensor(sen[1:], dtype=torch.long).to(device))

# pad sequences
x_train = pad_sequence(x_train, padding_value=pad, batch_first=True)
y_train = pad_sequence(y_train, padding_value=pad, batch_first=True)

# select train and test sentences
if not args.pre:
	index = np.arange(x_train.shape[0])
	train_index, test_index = train_test_split(index, test_size=0.2, shuffle=True)
	np.save("train_index", train_index)
	np.save("test_index", test_index)
else:
	train_index = np.load("train_index.npy")
	test_index = np.load("test_index.npy")

x_test = x_train[test_index]
x_train = x_train[train_index]
y_test = y_train[test_index]
y_train = y_train[train_index]

# sort samples by inverse number of zeros (padded inputs)
nz = (x_train==pad).sum(dim=1)
_, ind = torch.sort(nz, descending=False)
x_train = x_train[ind]
y_train = y_train[ind]

nz = (x_test==pad).sum(dim=1)
_, ind = torch.sort(nz, descending=False)
x_test = x_test[ind]
y_test = y_test[ind]

# make dataset and dataloader
_, samples_length = x_train.shape

train_input_lengths = samples_length - (x_train==pad).sum(dim=1)
test_input_lengths = samples_length - (x_test==pad).sum(dim=1)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train, train_input_lengths)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test, test_input_lengths)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# network configuration

embedding_dim = 100
hidden_dim = 100
nlayers = 2
clipping_value = 1
patience = 10

model = LSTM(embedding_dim, hidden_dim, nlayers, nlabels, samples_length=samples_length).to(device)
model.load_state_dict(torch.load("models/lstm_pin.pth")) if args.pre else 0
cel = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=pad)#weight=weights)
bce = torch.nn.BCELoss()
msel = torch.nn.MSELoss()
wd = 0.
optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=wd)

best_loss = np.inf
for epoch in range(args.epochs):
	model.train()
	train_loss = 0
	ti = time.time()
	for idx, (batch, y_true, batch_lengths) in enumerate(tqdm(train_loader)):
		optimizer.zero_grad()
		_, _, _, clf = model(batch, batch_lengths)
		y_true = y_true.view(-1)
		clf = clf.view(-1, nlabels)
		clf = clf[y_true!=pad, :]
		loss = cel(clf, y_true[y_true!=pad])
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
		optimizer.step()
		train_loss += loss.item()
	train_loss /= (idx + 1)
	test_loss = 0
	model.eval()
	with torch.no_grad():
		for idx, (batch, y_true, batch_lengths) in enumerate(tqdm(test_loader)):
			_, _, _, clf = model(batch, batch_lengths)
			y_true = y_true.view(-1)
			clf = clf.view(-1, nlabels)
			clf = clf[y_true!=pad, :]
			loss = cel(clf, y_true[y_true!=pad])
			test_loss += loss.item()
	test_loss /= (idx + 1)
	tf = time.time()
	print("Epoch {:03d} | Train loss {:.3f} | Test loss {:.3f} | Time {:.2f} min.".format(epoch, train_loss, test_loss, (tf - ti)/60))
	if test_loss<best_loss:
		print("Saving")
		torch.save(model.state_dict(), "models/lstm_pin.pth")
		best_loss = test_loss
		best_epoch = epoch
		patience = 10
	else:
		patience -= 1
	break if patience == 0 else 0

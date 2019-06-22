import numpy as np
from sklearn.preprocessing import OneHotEncoder
import argparse
import time
from tqdm import tqdm

import torch
import torch.utils.data
import torchvision

from Models import *
from utils import *


seed = 1111
seed_everything(seed)

parser = argparse.ArgumentParser(description="pineraBot")
parser.add_argument('--bs', type=int, default=100, metavar='B', help='input batch size for training (default: 128)')
parser.add_argument('--e', type=int, default=2, metavar='E', help='number of epochs to train (default: 10)')
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--lr", type=float, default=2e-4, metavar="L", help="learning rate")
parser.add_argument("--pre", action="store_true", help="train pre trained model (default False)")
parser.add_argument("--debug", action="store_true", help="enables debug (default False")
args = parser.parse_args()

device = args.d
print(device)

labels = np.load("labels.npy")
nlabels = len(labels)
eos = int(np.nonzero(labels=="<EOS>")[0][0])
pad = int(np.nonzero(labels=="<PAD>")[0][0])
num = int(np.nonzero(labels=="<NUM>")[0][0])

x_train = np.load("x_train.npy")
x_test = np.load("x_test.npy")
x_val = np.load("x_val.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
y_val = np.load("y_val.npy")

x_train = torch.tensor(x_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.long)
x_val = torch.tensor(x_val, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

x_train, y_train = sort_by_zeros(x_train, y_train)
x_test, y_test = sort_by_zeros(x_test, y_test)
x_val, y_val = sort_by_zeros(x_val, y_val)

_, samples_length = x_train.shape

train_input_lengths = compute_lengths(x_train)
test_input_lengths = compute_lengths(x_test)
val_input_lengths = compute_lengths(x_val)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train, train_input_lengths)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test, test_input_lengths)
val_dataset = torch.utils.data.TensorDataset(x_val, y_val, val_input_lengths)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

# network configuration
if args.debug:
	embedding_dim = 2
	hidden_dim = 2
	nlayers = 1
else:
	embedding_dim = 100
	hidden_dim = 100
	nlayers = 2
clipping_value = 1

model = LSTM(embedding_dim, hidden_dim, nlayers, nlabels, samples_length=samples_length).to(device)
model.load_state_dict(torch.load("models/lstm_bot.pth")) if args.pre else 0
print("Parameters: {}".format(count_parameters(model)))
cel = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad)
wd = 0.
optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=wd)
transform = torchvision.transforms.Compose([ToDevice(device)])

best_loss = np.inf
for epoch in range(args.e):
	model.train()
	train_loss = 0
	ti = time.time()
	for idx, (batch, y_true, batch_lengths) in enumerate(tqdm(train_loader)):
		batch, y_true, batch_lengths = transform([batch, y_true, batch_lengths])
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
		torch.save(model.state_dict(), "models/lstm_bot.pth")
		best_loss = test_loss
		best_epoch = epoch

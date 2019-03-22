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


class LSTM(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size):
		super(LSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)
		self.hidden2logit = nn.Linear(hidden_dim, vocab_size)


	def forward(self, sentence, input_lengths):
		embd = self.word_embeddings(sentence)
		packed_input = pack_padded_sequence(embd, input_lengths, batch_first=True)
		packed_output, (h_n, c_n) = self.lstm(packed_input)
		output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=samples_length)		
		logit = self.hidden2logit(output)
		return output, h_n, c_n, logit


def myFunc(e):
  return len(e)

#seed = 1111
#seed_everything(seed)
parser = argparse.ArgumentParser(description='PetFinder')
parser.add_argument('--batch_size', type=int, default=100, metavar='B', help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs to train (default: 10)')
parser.add_argument('--cuda', type=int, default=1, help='enables CUDA training')
parser.add_argument('--show_each', type=int, default=5, metavar='SE', help='show train and test stats each n epochs')
parser.add_argument('--plot', type=bool, default=False, metavar='P', help='plot train and test stats each n epochs')
parser.add_argument('--competition', type=bool, default=False, metavar='C', help="code for competition or not")
parser.add_argument("--lr", type=float, default=1e-3, metavar="L", help="learning rate")
args = parser.parse_args()

device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
print(device)

labels = np.load("labels.npy")
nlabels = len(labels)
sentences = np.load("num_sentences.npy")
x_train = []
y_train = []
lengths = []
for i, sen in enumerate(sentences):
	#print("{}/{}".format(i, len(sentences)))
	if (sen==0).sum()>0:
		sen[sen==0] = nlabels
	lengths.append(len(sen))
	x_train.append(torch.tensor(sen[:-1], dtype=torch.long).to(device))
	y_train.append(torch.tensor(sen[1:], dtype=torch.long).to(device))

x_train.sort(key=myFunc, reverse=True)
y_train.sort(key=myFunc, reverse=True)
x_train = pad_sequence(x_train, batch_first=True)
y_train = pad_sequence(y_train, batch_first=True)

n_samples, samples_length = x_train.shape
input_lengths = samples_length - (x_train==0).sum(dim=1)


train_dataset = torch.utils.data.TensorDataset(x_train, y_train, input_lengths)
#test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

# make dataloader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
#pdb.set_trace()
embedding_dim = 100
hidden_dim = 10

model = LSTM(embedding_dim, hidden_dim, len(labels)).to(device)
model.load_state_dict(torch.load("models/lstm_pin.pth"))
cel = torch.nn.CrossEntropyLoss(ignore_index=0)#weight=weights)
bce = torch.nn.BCELoss()
msel = torch.nn.MSELoss()
wd = 0.
optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=wd)
#optimizer = optim.SGD(model.parameters(), lr=0.1)

best_loss = np.inf
for epoch in range(args.epochs):
	model.train()
	train_loss = 0
	ti = time.time()
	for idx, (batch, y_true, batch_lengths) in enumerate(train_loader):
		model.zero_grad()
		output, hn, cn, clf = model(batch, batch_lengths)
		clf = clf.transpose(1, 2)
		loss = cel(clf, y_true)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
	tf = time.time()
	print("Epoch {} Loss {:.2f} Time {:.2f}".format(epoch, train_loss/(idx + 1), (tf - ti)/60))
	if train_loss<best_loss:
		print("Saving")
		torch.save(model.state_dict(), "models/lstm_pin.pth")
		best_loss = train_loss
		best_epoch = epoch
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
from Models import *
from creds import twitterUser


def vec2word(x, labels):
	sen = ""
	words = []
	for i in x:
		words.append(labels[i]) 
	s = " "
	return s.join(words)


def finalProcess(x):
	x = x[0].upper() + x[1:] + "."
	return x


parser = argparse.ArgumentParser(description="pineraBot")
parser.add_argument("--cuda", type=bool, default=1, help="enables CUDA training (default 1)")
parser.add_argument("--tweet", type=bool, default=0, help="enables tweeting (default 0)")
args = parser.parse_args()


device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
print(device)
user = twitterUser()

labels = np.load("labels.npy")
nlabels = len(labels)

embedding_dim = 100
hidden_dim = 100
nh = 200
eos = np.nonzero(labels=="eos")[0][0]
x_init_numpy = np.random.randint(0, nlabels)
x_init = torch.tensor(x_init_numpy).to(device)

softmax = torch.nn.Softmax(dim=0)

model = LSTM(embedding_dim, hidden_dim, nh, nlabels + 1).to(device)

model.load_state_dict(torch.load("models/lstm_pin.pth"))
sentence = []
test_sentence = ""
maxwords = 20
with torch.no_grad():
	model.eval()
	while len(test_sentence)<119:
		y = model.word_embeddings(x_init)
		output, (h_n, c_n) = model.lstm(y.view(1, 1, -1))
		prob = softmax(model.out(output.squeeze())).cpu().numpy()
		next_word = np.random.choice(np.arange(len(prob)), p=prob)
		if next_word==eos:
			break
		else:
			sentence.append(next_word)
			test_sentence = vec2word(sentence, labels)
			x_init = torch.tensor(next_word).long().to(device)
gen_sentence = vec2word(sentence, labels)
gen_sentence = finalProcess(gen_sentence)
user.tweet(gen_sentence) if args.tweet else print(gen_sentence)

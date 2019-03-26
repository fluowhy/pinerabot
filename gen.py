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


def vec2word(x, labels):
	sen = ""
	words = []
	for i in x:
		words.append(labels[i]) 
	s = " "
	return s.join(words)


def finalProcess(x):
	x = x[0].upper() + x[1:] + "."
	x = x.replace(" , ", ",").replace(" : ", ":").replace(" ? ", "?").replace(" ¿ ", "¿")
	x = x.replace(" ,, ", "\"").replace(" ; ", ";").replace(" ! ", "!").replace(" ¡ ", "¡")
	x = x.replace(" ,, ", "\'").replace(" ( ", "(").replace(" ) ", ")").replace(" - ", "-")
	x = x.replace(" ... ", "...").replace(" % ", "%").replace(" ***** ", "*****")
	x = x.replace("<NUM>", str(np.random.randint(0, 100)))
	return x


parser = argparse.ArgumentParser(description="pineraBot")
parser.add_argument("--cuda", action="store_true", help="enables CUDA training (default False)")
parser.add_argument("--tweet", action="store_true", help="enables tweeting (default False)")
args = parser.parse_args()

device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
print(device)

labels = np.load("labels.npy")
nlabels = len(labels)

embedding_dim = 100
hidden_dim = 100
eos = int(np.nonzero(labels=="<EOS>")[0][0])
pad = int(np.nonzero(labels=="<PAD>")[0][0])
num = int(np.nonzero(labels=="<NUM>")[0][0])
x_init_numpy = np.random.randint(0, nlabels)
x_init = torch.tensor(x_init_numpy).to(device)

softmax = torch.nn.Softmax(dim=0)

model = LSTM(embedding_dim, hidden_dim, nlabels).to(device)

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

if args.tweet:
	from creds import twitterUser
	user = twitterUser()
	user.followFollowers()
	user.tweet(gen_sentence)
else:
	print(gen_sentence)

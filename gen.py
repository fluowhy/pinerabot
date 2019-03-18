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


class LSTM(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size):
		super(LSTM, self).__init__()
		self.hidden_dim = hidden_dim

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)

		self.hidden2tag = nn.Linear(hidden_dim, vocab_size)
		

	def forward(self, sentence, input_lengths):
		embd = self.word_embeddings(sentence)
		# embedding works with padded samples
		# pack embedded samples
		packed_input = pack_padded_sequence(embd, input_lengths, batch_first=True)
		packed_output, (h_n, c_n) = self.lstm(packed_input)
		output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=samples_length)		
		tag_space = self.hidden2tag(output)#output.view(len(sentence), -1))
		#tag_scores = F.log_softmax(tag_space, dim=1)
		return output, h_n, c_n, tag_space


device = "cuda"

labels = np.load("labels.npy")
nlabels = len(labels)

embedding_dim = 100
hidden_dim = 10
eos = np.nonzero(labels=="eos")[0][0]
print("eos {}".format(eos))
x_init_numpy = np.random.randint(0, nlabels)
x_init = torch.tensor(x_init_numpy).to(device)

model = LSTM(embedding_dim, hidden_dim, nlabels).to(device)

model.load_state_dict(torch.load("models/lstm_pin.pth"))
sentence = []
with torch.no_grad():
	model.eval()
	while True:
		print(x_init)
		y = model.word_embeddings(x_init)
		output, (h_n, c_n) = model.lstm(y.view(1, 1, -1))
		print(h_n)
		next_x = np.argmax(model.hidden2tag(output).squeeze().cpu().numpy())
		if next_x==eos:
			break
		else:
			sentence.append(next_x)
			x_init = torch.tensor(next_x).to(device)
		print(sentence)
		input("hola")

pdb.set_trace()

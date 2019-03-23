import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import pdb


class LSTM(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, nh, vocab_size, samples_length=5):
		super(LSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)
		self.fc1 = nn.Linear(hidden_dim, nh)
		self.fc2 = nn.Linear(nh, vocab_size)
		self.samples_length = samples_length
		layers = []
		layers.append(torch.nn.Linear(hidden_dim, vocab_size))
		#layers.append(torch.nn.BatchNorm1d(samples_length))
		#layers.append(torch.nn.ReLU())
		#layers.append(torch.nn.Linear(nh, vocab_size))
		self.out = torch.nn.Sequential(*layers)


	def forward(self, sentence, input_lengths):
		embd = self.word_embeddings(sentence)
		packed_input = pack_padded_sequence(embd, input_lengths, batch_first=True)
		packed_output, (h_n, c_n) = self.lstm(packed_input)
		output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=self.samples_length)		
		logit = self.out(output)
		return output, h_n, c_n, logit

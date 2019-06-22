import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import pdb


class LSTM(torch.nn.Module):
	def __init__(self, embedding_dim, hidden_dim, nlayers, vocab_size, samples_length=5):
		super(LSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.word_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
		self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=nlayers)
		self.samples_length = samples_length
		self.out = torch.nn.Linear(hidden_dim, vocab_size)


	def forward(self, sentence, input_lengths):
		embd = self.word_embeddings(sentence)
		packed_input = pack_padded_sequence(embd, input_lengths, batch_first=True)
		packed_output, (h_n, c_n) = self.lstm(packed_input)
		output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=self.samples_length)		
		logit = self.out(output)
		return output, h_n, c_n, logit

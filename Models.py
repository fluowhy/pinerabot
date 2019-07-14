import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import pdb


class LSTM(torch.nn.Module):
	def __init__(self, nin, nhid, nlayers, nout, samples_length=1):
		super(LSTM, self).__init__()
		self.hidden_dim = nhid
		self.lstm = torch.nn.LSTM(input_size=nin, hidden_size=nhid, num_layers=nlayers)
		self.samples_length = samples_length
		self.out = torch.nn.Linear(nhid, nout)


	def forward(self, x, input_lengths):
		packed_input = pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
		packed_output, (_, _) = self.lstm(packed_input)
		output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=self.samples_length)
		y_pred = self.out(output)
		return y_pred

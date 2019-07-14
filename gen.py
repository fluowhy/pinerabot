import pandas as pd
import torch.nn as nn
import argparse
import string
from sklearn.externals import joblib

from Models import *
from utils import *


def vec2word(x, encoder):
	sentence = encoder.inverse_transform(x)
	s = ""
	return s.join(sentence)


def finalProcess(x):
	x = x[0].upper() + x[1:] + "."
	x = x.replace(" , ", ",").replace(" : ", ":").replace(" ? ", "?").replace(" ¿ ", "¿")
	x = x.replace(" ,, ", "\"").replace(" ; ", ";").replace(" ! ", "!").replace(" ¡ ", "¡")
	x = x.replace(" ,, ", "\'").replace(" ( ", "(").replace(" ) ", ")").replace(" - ", "-")
	x = x.replace(" ... ", "...").replace(" % ", "%").replace(" ***** ", "*****")
	x = x.replace("<NUM>", str(np.random.randint(0, 100)))
	return x


parser = argparse.ArgumentParser(description="pineraBot")
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--tweet", action="store_true", help="enables tweeting (default False)")
args = parser.parse_args()

device = args.d
print(device)

char_labels = np.load("labels.npy")
labels = np.load("encoded_labels.npy")
nlabels = len(labels)

params = pd.read_csv("params.csv")

nin = int(params["nin"].values[0])
hidden_dim = int(params["nhid"].values[0])
nlayers = int(params["nlayers"].values[0])
max_length = 280
label2onehot = Label2OneHot(nlabels + 1)
le = joblib.load("label_encoder")

l = list(string.ascii_letters.upper())
x_init = np.random.choice(l, 1)
x_init_label = le.transform(np.array([x_init]))
x_init_torch = torch.tensor(x_init_label, dtype=torch.uint8, device=device)
x_one_hot = label2onehot(x_init_torch).unsqueeze(0)

eos = le.transform(np.array(["."]))[0]

softmax = torch.nn.Softmax(dim=0)

model = LSTM(nin, hidden_dim, nlayers, nin, max_length)
model.load_state_dict(torch.load("models/lstm_bot.pth", map_location=args.d))
model.to(device)
sentence = []
test_sentence = ""


model.eval()
sentence.append(x_init_label.item())
with torch.no_grad():
	for i in range(max_length):
		input_lengths = torch.tensor([i + 1], dtype=torch.long, device=device)
		y_pred = model(x_one_hot, input_lengths)
		probs = softmax(y_pred[0, 0, 1:]).cpu().numpy()
		next_letter = np.random.choice(np.arange(len(probs)), p=probs)
		sentence.append(next_letter)
		if next_letter == eos:
			break
		else:
			x_label = torch.tensor(sentence, dtype=torch.uint8, device=device)
			x_one_hot = label2onehot(x_label).unsqueeze(0)
		f = 0
gen_sentence = vec2word(sentence, le)

if args.tweet:
	from creds import twitterUser
	user = twitterUser()
	user.followFollowers()
	user.tweet(gen_sentence)
else:
	0
print(gen_sentence)

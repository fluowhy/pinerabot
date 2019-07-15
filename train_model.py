import argparse
from tqdm import tqdm
import pandas as pd
import torch.utils.data
import torchvision

from Models import *
from utils import *


def check_and_remove(x, y, x_length):
	# check and remove zero length sentences
	lenght = (x_length == 0).sum().item()
	if lenght != 0:
		not_zero_lenght = x_length != 0
		return x[not_zero_lenght], y[not_zero_lenght], x_length[not_zero_lenght]
	else:
		return x, y, x_length


seed = 1111
seed_everything(seed)
dpi = 400

parser = argparse.ArgumentParser(description="pineraBot")
parser.add_argument('--bs', type=int, default=30, help='input batch size for training (default: 128)')
parser.add_argument('--e', type=int, default=2, help='number of epochs to train (default: 10)')
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default 2e-4)")
parser.add_argument("--pre", action="store_true", help="train pre trained model (default False)")
parser.add_argument("--debug", action="store_true", help="enables debug (default False")
parser.add_argument('--hs', type=int, default=2, help="hidden size (default: 2)")
args = parser.parse_args()

device = args.d
print(device)

labels = np.load("encoded_labels.npy")
nlabels = len(labels)

x_train = np.load("x_train.npy")
x_test = np.load("x_test.npy")
x_val = np.load("x_val.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
y_val = np.load("y_val.npy")

if args.debug:
	x_train = x_train[:20]
	x_test = x_test[:20]
	x_val = x_val[:20]
	y_train = y_train[:20]
	y_test = y_test[:20]
	y_val = y_val[:20]

x_train = torch.tensor(x_train, dtype=torch.uint8, device="cpu")
x_test = torch.tensor(x_test, dtype=torch.uint8, device="cpu")
x_val = torch.tensor(x_val, dtype=torch.uint8, device="cpu")
y_train = torch.tensor(y_train, dtype=torch.uint8, device="cpu")
y_test = torch.tensor(y_test, dtype=torch.uint8, device="cpu")
y_val = torch.tensor(y_val, dtype=torch.uint8, device="cpu")

x_train, y_train = sort_by_zeros(x_train, y_train)
x_test, y_test = sort_by_zeros(x_test, y_test)
x_val, y_val = sort_by_zeros(x_val, y_val)

_, samples_length = x_train.shape

train_input_lengths = compute_lengths(x_train).type(dtype=torch.uint8)
test_input_lengths = compute_lengths(x_test).type(dtype=torch.uint8)
val_input_lengths = compute_lengths(x_val).type(dtype=torch.uint8)

x_train, y_train, train_input_lengths = check_and_remove(x_train, y_train, train_input_lengths)
x_test, y_test, test_input_lengths = check_and_remove(x_test, y_test, test_input_lengths)
x_val, y_val, val_input_lengths = check_and_remove(x_val, y_val, val_input_lengths)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train, train_input_lengths)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test, test_input_lengths)
val_dataset = torch.utils.data.TensorDataset(x_val, y_val, val_input_lengths)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

# network configuration
if args.debug:
	nin = nlabels + 1
	hidden_dim = args.hs
	nlayers = 1
else:
	nin = nlabels + 1
	hidden_dim = args.hs
	nlayers = 2
clipping_value = 1

# save hiperparameters

df = {"nin": [nin], "nhid": [hidden_dim], "nlayers": [nlayers]}
df = pd.DataFrame(data=df)
df.to_csv("params.csv", index=False)

model = LSTM(nin, hidden_dim, nlayers, nin, samples_length=samples_length)
model.load_state_dict(torch.load("models/lstm_bot.pth")) if args.pre else 0
model.to(device)
print("Parameters: {}".format(count_parameters(model)))
cel = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=0).to(device)
wd = 0.
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
transform = torchvision.transforms.Compose([ToDevice(device)])
label2onehot = Label2OneHot(nlabels + 1)
tolong = ToLong()

def train_my_model(model, optimizer, dataloader, clipping_value):
	model.train()
	train_loss = 0
	for idx, (batch, y_true, batch_lengths) in enumerate(tqdm(dataloader)):
		batch, y_true, batch_lengths = transform([batch, y_true, batch_lengths])
		batch = label2onehot(batch)
		y_true = tolong(y_true)
		batch_lengths = tolong(batch_lengths)
		optimizer.zero_grad()
		y_pred = model(batch, batch_lengths).transpose(2, 1)
		loss = cel(y_pred, y_true).sum(1).mean()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
		optimizer.step()
		train_loss += loss.item()
	train_loss /= (idx + 1)
	return train_loss


def eval_my_model(model, dataloader):
	eval_loss = 0
	model.eval()
	with torch.no_grad():
		for idx, (batch, y_true, batch_lengths) in enumerate(tqdm(dataloader)):
			batch, y_true, batch_lengths = transform([batch, y_true, batch_lengths])
			batch = label2onehot(batch)
			y_true = tolong(y_true)
			batch_lengths = tolong(batch_lengths)
			y_pred = model(batch, batch_lengths).transpose(2, 1)
			loss = cel(y_pred, y_true).sum(1).mean()
			eval_loss += loss.item()
	eval_loss /= (idx + 1)
	return eval_loss

losses = np.zeros((args.e, 3))
best_loss = np.inf
for epoch in range(args.e):
	train_loss = train_my_model(model, optimizer, train_loader, clipping_value)
	test_loss = eval_my_model(model, test_loader)
	val_loss = eval_my_model(model, val_loader)
	losses[epoch] = [train_loss, test_loss, val_loss]
	print("Epoch {:03d} | Train loss {:.3f} | Test loss {:.3f} | Val loss {:.3f}".format(epoch, train_loss, test_loss, val_loss))
	if val_loss < best_loss:
		print("Saving")
		torch.save(model.state_dict(), "models/lstm_bot.pth")
		best_loss = val_loss
	np.save("losses", losses)

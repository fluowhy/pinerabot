import numpy as np
from sklearn.preprocessing import LabelEncoder
import unidecode
import pdb
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def cleanText(x):
	x = x.replace("\r", "").replace("\n", "").replace("\xa0", "")
	return [xx for xx in map(str.strip, x.split('.')) if xx]


def cleanSentence(x):
	c = []
	clean_x = unidecode.unidecode(x).lower()
	clean_x = clean_x.replace(",", " , ").replace(":", " : ").replace("?", " ? ").replace("¿", " ¿ ")
	clean_x = clean_x.replace("\"", " ,, ").replace(";", " ; ").replace("!", " ! ").replace("¡", " ¡ ")
	clean_x = clean_x.replace("\'", " ,, ").replace("(", " ( ").replace(")", " ) ").replace("-", " - ")
	clean_x = clean_x.replace("...", ' ... ').replace("%", " % ").replace("*****", " ***** ")
	for i in clean_x.split(" "):
		c.append(num) if bool(re.search(r"\d", i)) else c.append(i)
	c.append(eos)
	return c


if __name__=="__main__":
	"""
	le = LabelEncoder()
	dis = np.load("discursos.npy")
	sen = []
	numsen = []
	print("{} speeches".format(len(dis)))
	unique = []
	pad = "<PAD>"
	eos = "<EOS>"
	num = "<NUM>"
	seqlen = []
	for i in dis:
		clean_speech = cleanText(i)
		for sentence in clean_speech:
			cleaned_sentence = cleanSentence(sentence)		
			sen.append(cleaned_sentence)
			unique += list(set(cleaned_sentence))
	unique = list(set(unique))
	unique.insert(0, pad)
	le.fit(unique)
	print("{} sentences".format(len(sen)))
	print("{} features".format(len(le.classes_)))
	np.save("labels", le.classes_)
	for i, ss in enumerate(tqdm(sen)):
		numsen.append(le.transform(ss))
		seqlen.append(len(ss))
	np.save("num_sentences", numsen)
	np.save("seqlen", seqlen)
	"""
	# pad sentences
	sentences = np.load("num_sentences.npy", allow_pickle=True)
	seqlen = np.load("seqlen.npy", allow_pickle=True)

	nsen = len(sentences)

	padded_sequence = np.zeros((nsen, max(seqlen)), dtype=np.int)
	for i, sen in tqdm(enumerate(sentences)):
		padded_sequence[i, :seqlen[i]] = sen
		f = 0
	np.save("padded_sentences", padded_sequence)

	# train val test split
	indexes = np.arange(nsen)
	train_idx, test_idx = train_test_split(indexes, test_size=0.2, shuffle=True)
	train_idx, val_idx = train_test_split(train_idx, test_size=0.1/0.8, shuffle=True)

	seqlen = np.array(seqlen)

	np.save("x_train", padded_sequence[train_idx])
	np.save("x_test", padded_sequence[test_idx])
	np.save("x_val", padded_sequence[val_idx])
	np.save("y_train", seqlen[train_idx])
	np.save("y_test", seqlen[test_idx])
	np.save("y_val", seqlen[val_idx])
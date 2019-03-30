import numpy as np
from sklearn.preprocessing import LabelEncoder
import unidecode
import pdb
import re
from tqdm import tqdm


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
	le = LabelEncoder()
	dis = np.load("discursos.npy")
	sen = []
	numsen = []
	print("{} speeches".format(len(dis)))
	unique = []
	pad = "<PAD>"
	eos = "<EOS>"
	num = "<NUM>"
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
		if i%1000==0:
			np.save("num_sentences", numsen)
	np.save("num_sentences", numsen)

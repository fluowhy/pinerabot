import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import unidecode
import pdb


def cleanText(x):
	x = x.replace("\r", "").replace("\n", "").replace("\xa0", "")
	return [xx for xx in map(str.strip, x.split('.')) if xx]


def cleanSentence(x):
	clean_x = unidecode.unidecode(x).lower()
	clean_x = clean_x.replace(",", " , ").replace(":", " : ").replace("?", " ? ").replace("¿", " ¿ ")
	clean_x = clean_x.replace("\"", " ,, ").replace(";", " ; ").replace("!", " ! ").replace("¡", " ¡ ")
	clean_x = clean_x.replace("\'", " ,, ").replace("(", " ( ").replace(")", " ) ").replace("-", " - ")
	clean_x = clean_x.replace("...", ' ... ').replace("%", " % ").replace("*****", " ***** ")
	clean_x += " eos "
	return clean_x


le = LabelEncoder()
dis = np.load("discursos.npy")
sen = []
numsen = []
print("{} speeches".format(len(dis)))
nsentences = 0
long_sentence = ""
for i in dis:
	clean_speech = cleanText(i)
	nsentences += len(clean_speech)
	for sentence in clean_speech:
		clean_sentence = cleanSentence(sentence)
		sen.append(clean_sentence)
		long_sentence += clean_sentence
le.fit(long_sentence.split())
print("{} sentences".format(nsentences))
print("{} features".format(len(le.classes_)))
np.save("labels", le.classes_)
for i, ss in enumerate(sen):
	print("{}/{}".format(i, nsentences))
	numsen.append(le.transform(ss.split()))
	if i%1000==0:
		np.save("num_sentences", numsen)
np.save("num_sentences", numsen)

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from tqdm import tqdm


sentences = np.load("processed_sentences.npy", allow_pickle=True)
nsentences = len(sentences)
labels = np.load("labels.npy")
le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(labels)
encoded_labels += 1
joblib.dump(le, "label_encoder")
encoded_sentences = []
seq_len = []
for sentence in tqdm(sentences):
    encoded_sentence = le.transform(sentence) + 1
    encoded_sentences.append(encoded_sentence)
    seq_len.append(len(sentence))

max_seq_len = max(seq_len)
padded_seq = np.zeros((nsentences, max_seq_len), dtype=np.uint8)
for idx, encoded_sentence in tqdm(enumerate(encoded_sentences)):
    padded_seq[idx, :seq_len[idx]] = encoded_sentence

input_seq = padded_seq[:, :-1]
target_seq = padded_seq[:, 1:]

indexes = np.arange(nsentences)
train_idx, test_idx = train_test_split(indexes, test_size=0.2, shuffle=True)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1/0.8, shuffle=True)

np.save("x_train", input_seq[train_idx])
np.save("x_test", input_seq[test_idx])
np.save("x_val", input_seq[val_idx])
np.save("y_train", target_seq[train_idx])
np.save("y_test", target_seq[test_idx])
np.save("y_val", target_seq[val_idx])

np.save("encoded_labels", encoded_labels)

f = 0
import numpy as np
import unidecode


delete = ["\n", "\r", "\t"]

discursos = np.load("discursos.npy")
processed_sentences = []
for i in range(len(discursos)):
    f = discursos[i]
    f = unidecode.unidecode(f)
    for d in delete:
        f = f.replace(d, "")
    f = f.replace("'", "\"")
    g = list(f)
    if i == 0:
        h = np.unique(g)
    else:
        h1 = np.unique(g)
        for j in h1:
            if (h == j).sum() == 0:
                h = np.append(h, j)
    # split speech by sentences
    sentences = f.split(".")[:-1]
    for sen in sentences:
        sen = list(sen)
        sen.append(".")
        processed_sentences.append(sen)
np.save("processed_sentences", processed_sentences)
np.save("labels", h)

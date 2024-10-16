import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from unicodedata import east_asian_width as eaw

# apt-get install fonts-nanum*
# fc-cache -fv
# python3 -c "import matplotlib; print(matplotlib.__path__)"
# cp /usr/share/fonts/truetype/nanum/Nanum* path/mpl-data/fonts/ttf/
# mpl.get_cachedir()
# rm fontlist.json

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.unicode_minus"] = False

def usl(x): # unicode string length

    return sum(2 if eaw(c) == "W" else 1 for c in x)

def cosine_similarity(x, y):

    return np.dot(x, y) / np.linalg.norm(x) * np.linalg.norm(y)

def normalize(x, axis, method):

    if method == "min-max":
        x -= x.min(axis = axis, keepdims = True)

    if method == "softmax":
        x = np.exp(x - x.max(axis = axis, keepdims = True))

    z = x.sum(axis = axis, keepdims = True)

    return np.divide(x, z, out = np.zeros_like(x), where = (z != 0))

class dataloader():

    def __init__(self, filename, batch_size):

        self.filename = filename
        self.data_size = 0
        self.batch_size = batch_size

    def batch(self):

        batch = []
        fo = open(self.filename)

        while True:
            line = fo.readline()
            if line:
                batch.append(line.strip())
                if len(batch) < self.batch_size:
                    continue
            if batch:
                self.data_size += len(batch)
                yield batch
                batch.clear()
            if not line:
                break

        fo.close()

def ngrams(tokens, maxlen):

    for i in range(len(tokens)):
        for j in range(i + 1, min(len(tokens), i + maxlen) + 1):
            w = re.sub("\\s+", " ", " ".join(tokens[i:j])).strip()
            yield (i, j), w

def img_alignment_map(m, xws, yws, threshold = 0.01):

    sns.heatmap(
        data = [[0 if x < threshold else x for x in y] for y in m],
        cmap = "Reds",
        cbar = False,
        xticklabels = yws,
        yticklabels = xws,
        annot = True
    )

    plt.show()

def txt_alignment_map(m, xws, yws, threshold = 0.01):

    xi = [str(i)[-1] for i in range(len(xws))]
    yi = [str(i)[-1] for i in range(len(yws))]

    xwl = max(map(usl, xws))
    ywl = max(map(len, yws))

    hl = "+" + "-" * (xwl + len(yws) * 2 + 4) + "+" # horizontal line
    nd = " " * xwl # indent

    xws = [" " * (xwl - usl(w)) + w for w in xws]
    yws = [[" " * (2 - usl(c)) + c for c in w + " " * (ywl - len(w))] for w in yws]

    print(hl)
    print("\n".join(" ".join(
        ["|", w, i, *["." if x < threshold else "*" for x in xs], "|"]
        ) for (w, i, xs) in zip(xws, xi, m)
    ))
    print(" ".join(["|", nd, " ", *yi, "|"]))
    print("\n".join(" ".join(["|", nd, "", "".join(cs), "|"]) for cs in zip(*yws)))
    print(hl)

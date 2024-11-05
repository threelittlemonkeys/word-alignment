import sys
import re
import time
from utils import *
from sentence_transformers import SentenceTransformer

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# suppress InsecureRequestWarning
import requests
requests.packages.urllib3.disable_warnings()

class phrase_aligner():

    def __init__(self, src_lang, tgt_lang, batch_size, window_size, thresholds, verbose):

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.window_size = window_size
        self.phrase_score_threshold = thresholds[0]
        self.alignment_score_threshold = thresholds[1]
        self.verbose = verbose

        print(f"src_lang = {src_lang}", file = sys.stderr)
        print(f"tgt_lang = {tgt_lang}", file = sys.stderr)
        print(f"batch_size = {batch_size}", file = sys.stderr)
        print(f"window_size = {window_size}", file = sys.stderr)
        print(f"phrase_score_threshold = {self.phrase_score_threshold}", file = sys.stderr)
        print(f"alignment_score_threshold = {self.alignment_score_threshold}", file = sys.stderr)

        self.model = self.load_model()

    def load_model(self):

        # Language-agnostic BERT Sentence Embedding (LaBSE)
        print("loading LaBSE", file = sys.stderr)
        model = SentenceTransformer("sentence-transformers/LaBSE")
        print("loaded LaBSE", file = sys.stderr)

        # if requests.exceptions.SSLError occurs:
        # add kwargs["verify"] = False
        # in def send(self, request: PreparedRequest, *args, **kwargs) -> Response:
        # in class UniqueRequestIdAdapter(HTTPAdapter):
        # in /lib/python/site-packages/huggingface_hub/utils/_http.py

        return model

    def preproc(self, batch):

        n = self.window_size - 1
        ps = []
        pad = [""] * n
        data = []

        for line in batch:

            x, y = line.split("\t")
            xws = pad + re.sub("\\s+", " ", x).strip().split(" ") + pad
            yws = pad + re.sub("\\s+", " ", y).strip().split(" ") + pad
            xrs, xps = zip(*ngrams(xws, self.window_size))
            yrs, yps = zip(*ngrams(yws, self.window_size))
            ps.extend(xps)
            ps.extend(yps)
            data.append(((x, xws, xrs, xps), (y, yws, yrs, yps)))

        i = 0
        hs = self.model.encode(ps)

        for (x, xws, xrs, xps), (y, yws, yrs, yps) in data:

            xhs = hs[i:i + len(xps)]
            i += len(xps)
            yhs = hs[i:i + len(yps)]
            i += len(yps)

            if self.verbose:

                print(f"\nsrc_text\t{x}")
                print(f"tgt_text\t{y}")
                print(f"src_tokens\t{xws[n:len(xws) - n]}")
                print(f"tgt_tokens\t{yws[n:len(yws) - n]}\n")

            yield xws, yws, (xrs, xps, xhs), (yrs, yps, yhs)

    def sentence_similarity(self, batch):

        sents = []

        for line in batch:
            x, y = line.split("\t")
            sents.extend([x, y])

        hs = self.model.encode(sents)

        for i in range(0, len(hs), 2):
            yield cosine_similarity(*hs[i: i + 2])

    def align_words(self, Wa, xws, yws):

        Wa_xy = normalize(Wa, axis = 1, method = "softmax")
        Wa_yx = normalize(Wa, axis = 0, method = "softmax")

        Wa_xy *= (Wa_xy >= self.alignment_score_threshold)
        Wa_yx *= (Wa_yx >= self.alignment_score_threshold)

        # Wa = Wa_xy * Wa_yx
        Wa = np.maximum(Wa_xy, Wa_yx)

        A_xy = {*zip(range(Wa.shape[0]), Wa.argmax(axis = 1))}
        A_yx = {*zip(Wa.argmax(axis = 0), range(Wa.shape[1]))}

        Aw = {
            ((i, i + 1), (j, j + 1)): (Wa[i][j], (xws[i], yws[j]))
            for i, j in (A_xy & A_yx)
        }

        if self.verbose:

            print("word_alignments =")
            for xyr, (alignment_score, xyp) in sorted(Aw.items()):
                print(f"{alignment_score:.4f} {xyr} {xyp}")
            print()

        return Wa, Aw

    def cluster_phrases(self, m):

        def _func(m, i, j, k):

            m[i][j] = k

            if i > 0 and m[i - 1][j] < 0:
                _func(m, i - 1, j, k)
            if j > 0 and m[i][j - 1] < 0:
                _func(m, i, j - 1, k)
            if i < m.shape[0] - 1 and m[i + 1][j] < 0:
                _func(m, i + 1, j, k)
            if j < m.shape[1] - 1 and m[i][j + 1] < 0:
                _func(m, i, j + 1, k)

        k = 1
        m *= -1

        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i][j] >= 0:
                    continue
                _func(m, i, j, k)
                k += 1

    def align_phrases(self, Wa, Aw, xys):

        Ma = (Wa > 0).astype(int)
        Ax = [None] * Wa.shape[0]
        Ay = [None] * Wa.shape[1]

        self.cluster_phrases(Ma)

        for xyr in Aw:
            Ax[xyr[0][0]] = xys[xyr]
            Ay[xyr[1][0]] = xys[xyr]

        for xy in sorted(xys.values())[::-1]:

            (x0, x1), (y0, y1) = xy[1]
            m = Ma[x0:x1, y0:y1]

            if xy[0] < self.phrase_score_threshold:
                continue

            if not m.sum():
                pass # continue

            if len({*m.flatten()} - {0}) != 1:
                continue

            # phrase boundary constraints

            if x0 > 0 and Ax[x0] == Ax[x0 - 1] != None \
            or y0 > 0 and Ay[y0] == Ay[y0 - 1] != None \
            or x1 < Wa.shape[0] - 1 and Ax[x1] == Ax[x1 - 1] != None \
            or y1 < Wa.shape[1] - 1 and Ay[y1] == Ay[y1 - 1] != None:
                continue

            _xys = {*Ax[x0:x1], *Ay[y0:y1]} - {None}

            if not _xys:
                continue

            _xrs, _yrs = zip(*[_xy[1] for _xy in _xys])
            _xrs, _yrs = [*zip(*_xrs)], [*zip(*_yrs)]
            _x0, _x1 = min(_xrs[0]), max(_xrs[1])
            _y0, _y1 = min(_yrs[0]), max(_yrs[1])

            if _x0 < x0 or x1 < _x1 or _y0 < y0 or y1 < _y1:
                continue

            Ax[x0:x1] = [xy] * (x1 - x0)
            Ay[y0:y1] = [xy] * (y1 - y0)

        Ap = {*Ax, *Ay} - {None}

        if self.verbose:
            print("phrase_alignments =")
            for xy in Ap:
                phrase_score, xyr, xyp = xy
                print(f"{phrase_score:.4f} {xyr} {xyp}")
            print()

        return Ap

    def align(self, xws, yws, xs, ys):

        # remove padding tokens

        n = self.window_size - 1
        xws = xws[n:len(xws) - n]
        yws = yws[n:len(yws) - n]

        Wa = np.zeros((len(xws), len(yws)))
        xys = {}

        # phrase similarity

        for xr, xp, xh in zip(*xs):
            xr = (xr[0] - n, xr[1] - n)
            for yr, yp, yh in zip(*ys):
                yr = (yr[0] - n, yr[1] - n)
                phrase_score = cosine_similarity(xh, yh)
                if 0 <= xr[0] and xr[1] <= len(xws) \
                and 0 <= yr[0] and yr[1] <= len(yws):
                    xys[(xr, yr)] = (phrase_score, (xr, yr), (xp, yp))
                if phrase_score >= self.phrase_score_threshold:
                    Wa[max(xr[0], 0):xr[1], max(yr[0], 0):yr[1]] += phrase_score

        # alignment

        Wa, Aw = self.align_words(Wa, xws, yws)
        Ap = self.align_phrases(Wa, Aw, xys)

        score = sum(
            (xr[1] - xr[0]) + (yr[1] - yr[0])
            for _, (xr, yr), _ in Ap
        ) / sum(Wa.shape)

        if self.verbose:
            print("alignment_map =")
            txt_alignment_map(Wa, xws, yws)
            # img_alignment_map(Wa, xws, yws)
            print()

        return score

if __name__ == "__main__":

    if len(sys.argv) not in (5, 6):
        sys.exit("Usage: %s src_lang tgt_lang phrase|sentence tokenized_bitext [-v]" % sys.argv[0])

    src_lang, tgt_lang, method, filename = sys.argv[1:5]

    aligner = phrase_aligner(
        src_lang = src_lang,
        tgt_lang = tgt_lang,
        batch_size = 1024,
        window_size = 3,
        thresholds = (0.7, 0.1),
        verbose = (len(sys.argv) == 6 and sys.argv[5] == "-v")
    )

    dataloader = dataloader(
        filename = filename,
        batch_size = aligner.batch_size
    )

    timer = time.time()

    for batch in dataloader.batch():

        if method == "phrase":

            for line, data in zip(batch, aligner.preproc(batch)):
                score = aligner.align(*data)
                print(score, line, sep = "\t")

                if aligner.verbose:
                    input()

        if method == "sentence":

            sentence_scores = aligner.sentence_similarity(batch)

            for line, sentence_similarity in zip(batch, sentence_scores):
                print(sentence_similarity, line, sep = "\t")

    print(f"{dataloader.data_size} lines ({time.time() - timer:.4f} seconds)", file = sys.stderr)
    timer = time.time()

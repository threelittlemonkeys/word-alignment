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

        ps = []
        pad = [""] * (self.window_size - 1)
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
                print(f"src_tokens\t{xws}")
                print(f"tgt_tokens\t{yws}\n")

            yield xws, yws, (xrs, xps, xhs), (yrs, yps, yhs)

    def sentence_similarity(self, batch):

        sents = []

        for line in batch:
            x, y = line.split("\t")
            sents.extend([x, y])

        hs = self.model.encode(sents)

        for i in range(0, len(hs), 2):
            yield cosine_similarity(*hs[i: i + 2])

    def alignment_score(self, Wa_xy, Wa_yx):

        Wa = Wa_xy * Wa_yx

        Wa_xy_argmax = {*zip(range(Wa.shape[0]), Wa.argmax(axis = 1))}
        Wa_yx_argmax = {*zip(Wa.argmax(axis = 0), range(Wa.shape[1]))}

        alignment_idxs = {
            idx for idx in Wa_xy_argmax & Wa_yx_argmax
            # if Wa[idx] >= self.alignment_score_threshold
        }

        for i, j in (*alignment_idxs,):
            for _Wa in (Wa_xy, Wa_yx):

                for k in range(i, -1, -1):
                    if _Wa[k][j] < self.alignment_score_threshold:
                        break
                    alignment_idxs.add((k, j))
                    Wa[k][j] = max(Wa[k][j], _Wa[k][j])

                for k in range(i, Wa.shape[0]):
                    if _Wa[k][j] < self.alignment_score_threshold:
                        break
                    alignment_idxs.add((k, j))
                    Wa[k][j] = max(Wa[k][j], _Wa[k][j])

                for k in range(j, -1, -1):
                    if _Wa[i][k] < self.alignment_score_threshold:
                        break
                    alignment_idxs.add((i, k))
                    Wa[i][k] = max(Wa[i][k], _Wa[i][k])

                for k in range(j, Wa.shape[1]):
                    if _Wa[i][k] < self.alignment_score_threshold:
                        break
                    alignment_idxs.add((i, k))
                    Wa[i][k] = max(Wa[i][k], _Wa[i][k])

        alignment_scores = Wa[*zip(*alignment_idxs)]
        alignment_score = sum(len(set(vs))
            for vs in [*zip(*[(i, j)
            for (i, j), v in zip(alignment_idxs, alignment_scores)
            if v >= self.alignment_score_threshold
        ])]) / (Wa.shape[0] + Wa.shape[1])

        return Wa, alignment_idxs, alignment_score

    def align(self, xws, yws, xs, ys):

        n = self.window_size - 1
        Wp = np.zeros((len(xws), len(yws)))
        xys = []

        for xr, xp, xh in zip(*xs):
            for yr, yp, yh in zip(*ys):
                phrase_score = cosine_similarity(xh, yh)
                u = (phrase_score, (xr, yr), (xp, yp))
                if phrase_score < self.phrase_score_threshold:
                    continue
                if n <= xr[0] < xr[1] <= len(xws) - n \
                and n <= yr[0] < yr[1] <= len(yws) - n:
                    xys.append(u)
                Wp[xr[0]:xr[1], yr[0]:yr[1]] += phrase_score

        Wp = Wp[n:-n, n:-n]
        xws = xws[n:-n]
        yws = yws[n:-n]

        Wa_xy = normalize(Wp, axis = 1, method = "softmax")
        Wa_yx = normalize(Wp, axis = 0, method = "softmax")

        Wa, alignment_idxs, alignment_score = self.alignment_score(Wa_xy, Wa_yx)

        img_alignment_map_args = ((Wa_xy, Wa_yx, Wa), xws, yws)

        if self.verbose:

            # print(f"Wa_xy = {Wa_xy.round(4).tolist()}")
            # print(f"Wa_yx = {Wa_yx.round(4).tolist()}")
            # print(f"Wa = {Wa.round(4).tolist()}\n")

            print("phrase_scores =")
            for phrase_score, (xr, yr), (xp, yp) in xys:
                print(f"{phrase_score:.4f} {(xr, yr)} {(xp, yp)}")
            print()

            print("alignment_scores =")
            for i, j in sorted(alignment_idxs):
                if Wa[i][j] < Wa[i][j]:
                    continue
                print(f"{Wa[i][j]:.4f} {(i, j)} {(xws[i], yws[j])}")

            print("\nalignment_map =")
            txt_alignment_map(Wa, yws, xws, self.alignment_score_threshold)
            print()

        return alignment_score, img_alignment_map_args

if __name__ == "__main__":

    if len(sys.argv) not in (5, 6):
        sys.exit("Usage: %s src_lang tgt_lang phrase|sentence tokenized_bitext [-v]" % sys.argv[0])

    src_lang, tgt_lang, method, filename = sys.argv[1:5]

    aligner = phrase_aligner(
        src_lang = src_lang,
        tgt_lang = tgt_lang,
        batch_size = 1024,
        window_size = 3,
        thresholds = (0.5, 0.1),
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
                alignment_score, img_alignment_map_args = aligner.align(*data)
                print(alignment_score, line, sep = "\t")

                if aligner.verbose:
                    img_alignment_map(*img_alignment_map_args)
                    input()

        if method == "sentence":

            sentence_scores = aligner.sentence_similarity(batch)

            for line, sentence_similarity in zip(batch, sentence_scores):
                print(sentence_similarity, line, sep = "\t")

    print(f"{dataloader.data_size} lines ({time.time() - timer:.4f} seconds)", file = sys.stderr)
    timer = time.time()

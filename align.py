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
                print(f"src_tokens\t{xws[n:-n]}")
                print(f"tgt_tokens\t{yws[n:-n]}\n")

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

        Wa = Wa_xy * Wa_yx
        Wa *= (Wa >= self.alignment_score_threshold)

        Wa_xy_idxs = {*zip(range(Wa.shape[0]), Wa.argmax(axis = 1))}
        Wa_yx_idxs = {*zip(Wa.argmax(axis = 0), range(Wa.shape[1]))}
        Wa_idxs = Wa_xy_idxs & Wa_yx_idxs
        Wa_score = len(Wa_idxs) * 2 / sum([*Wa.shape])

        if self.verbose:
            print("word_alignment_scores =")
            for i, j in sorted(Wa_idxs):
                print(f"{Wa[i][j]:.4f} {(i, j)} {(xws[i], yws[j])}")
            print()

        return Wa, Wa_idxs, Wa_score

    def align_phrases(self, Wa, xys):

        Wp = []
        Wp_idxs = [[-1] * Wa.shape[0], [-1] * Wa.shape[1]]

        for xy in sorted(xys)[::-1]:

            _, ((x0, x1), (y0, y1)), _ = xy
            A = (Wa[x0:x1, y0:y1] >= self.alignment_score_threshold).astype(int)

            if not A.sum():
                continue

            # phrase boundary constraints

            if x0 > 0 and 0 <= Wp_idxs[0][x0 - 1] == Wp_idxs[0][x0] \
            or y0 > 0 and 0 <= Wp_idxs[1][y0 - 1] == Wp_idxs[1][y0] \
            or x1 < Wa.shape[0] - 1 and 0 <= Wp_idxs[0][x1 - 1] == Wp_idxs[0][x1] \
            or y1 < Wa.shape[1] - 1 and 0 <= Wp_idxs[1][y1 - 1] == Wp_idxs[1][y1]:
                continue

            Wp_idxs[0][x0:x1] = [len(Wp)] * (x1 - x0)
            Wp_idxs[1][y0:y1] = [len(Wp)] * (y1 - y0)
            Wp.append(xy)

        if self.verbose:
            print("phrase_scores =")
            for phrase_score, (xr, yr), (xp, yp) in xys:
                print(f"{phrase_score:.4f} {(xr, yr)} {(xp, yp)}")
            print()

    def align(self, xws, yws, xs, ys):

        # remove padding tokens

        n = self.window_size - 1
        xws = xws[n:-n]
        yws = yws[n:-n]

        xys = []
        Wa = np.zeros((len(xws), len(yws)))

        # phrase similarity

        for xr, xp, xh in zip(*xs):
            xr = (xr[0] - n, xr[1] - n)
            for yr, yp, yh in zip(*ys):
                yr = (yr[0] - n, yr[1] - n)
                phrase_score = cosine_similarity(xh, yh)
                if phrase_score < self.phrase_score_threshold:
                    continue
                u = (phrase_score, (xr, yr), (xp, yp))
                if 0 <= xr[0] < xr[1] <= len(xws) \
                and 0 <= yr[0] < yr[1] <= len(yws):
                    xys.append(u)
                Wa[xr[0]:xr[1], yr[0]:yr[1]] += phrase_score

        # word alignment

        Wa, Wa_idxs, Wa_score = self.align_words(Wa, xws, yws)

        # phrase alignment

        self.align_phrases(Wa, xys)

        if self.verbose:
            print("\nalignment_map =")
            txt_alignment_map(Wa, xws, yws)
            # img_alignment_map(Wa, xws, yws)

        return Wa_score

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

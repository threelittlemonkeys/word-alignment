import sys
import re
import MeCab

# pip install mecab-python3
# pip install unidic-lite

tagger = MeCab.Tagger("-O wakati")

for line in sys.stdin:

    out = tagger.parse(line)
    out = re.sub("\\s+", " ", out).strip()
    print(out)

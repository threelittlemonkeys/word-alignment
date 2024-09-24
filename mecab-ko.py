import sys
import re
from mecab import MeCab

# pip install python-mecab-ko

mecab = MeCab()

for line in sys.stdin:

    out = mecab.morphs(line)
    out = re.sub("\\s+" , " ", " ".join(out)).strip()
    print(out)

#!/usr/bin/env python3

import sys
from data import spm_train_from_json

fname_in = sys.argv[1]
prefix = sys.argv[2]
if len(sys.argv) > 3:
    fname_tmp = sys.argv[3]
    if len(sys.argv) > 4:
        max_sents = int(sys.argv[4])
    else:
        max_sents = None
else:
    fname_tmp = './spm.tmp'
    max_sents = None

spm_train_from_json(fname_in, prefix, fname_tmp=fname_tmp, max_sents=max_sents)

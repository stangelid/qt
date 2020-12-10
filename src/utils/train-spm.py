#!/usr/bin/env python3

import sys
from data import spm_train_from_json

if len(sys.argv) < 3 or sys.argv[1][:2] == '-h':
    print('usage: python3 train-spm <json_file> <prefix> [<max_sentences>]')
    sys.exit()

fname_tmp = './spm.tmp'
fname_in = sys.argv[1]
prefix = sys.argv[2]
if len(sys.argv) > 3:
    max_sents = int(sys.argv[3])
else:
    max_sents = None

spm_train_from_json(fname_in, prefix, fname_tmp=fname_tmp, max_sents=max_sents)

#!/usr/bin/env python3

import sys
import os
import os.path
import json
from nltk.tokenize import sent_tokenize

if len(sys.argv) < 3 or sys.argv[1][:2] == '-h':
    print('usage: python3 json-to-dirs.py <json_file> <root_dir>')

jsonpath = sys.argv[1]
rootdir = sys.argv[2]

fjson = open(jsonpath, 'r')
data = json.load(fjson)
fjson.close()

for entity_data in data:
    entity_id = entity_data['entity_id']

    for summary_type, summaries in entity_data['summaries'].items():
        os.makedirs(os.path.join(rootdir, summary_type), exist_ok=True)
        for i, summary in enumerate(summaries):
            fname = os.path.join(rootdir, summary_type,
                    '{0}_{1}.txt'.format(entity_id, i))
            fout = open(fname, 'w')
            fout.write('\t'.join(sent_tokenize(summary)))
            fout.close()

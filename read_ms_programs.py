# parse programs from https://msr-redmond.github.io/karel-dataset/

import sys
import json
from tqdm import tqdm

#if len(sys.argv) < 2:
#    print("Need argument")
#    sys.exit(1)

MS_DATA_DIR = 'data_ms'

for name in ['train', 'test', 'val']:
    in_fname = '{}.json'.format(name)
    out_fname = '{}.prog'.format(name)

    with open(in_fname) as fr, open(out_fname, 'w') as fw:
        for line in tqdm(fr, total=1116854):
            j = json.loads(line)
            code =  " ".join(j['program_tokens'])
            fw.write(code + '\n')

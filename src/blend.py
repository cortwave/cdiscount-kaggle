from glob import glob
from collections import Counter

import pandas as pd
from tqdm import tqdm
from fire import Fire


def main(mask):
    files = filter(lambda x: not 'blend' in x, glob(f'results/*{mask}*csv*'))
    acc = {}

    for f in files:
        df = pd.read_csv(f)
        for id_, cat in tqdm(zip(df['_id'], df['category_id']), desc=f, total=df.shape[0]):
            if id_ in acc.keys():
                acc[id_].append(cat)
            else:
                acc[id_] = [cat]

    acc = [{'_id': k, 'category_id': Counter(v).most_common(1)[0][0]}
           for k, v in tqdm(acc.items(), desc='voting')]
    df = pd.DataFrame(acc)
    df.to_csv(f'results/blended_{mask}.csv.gzip', index=False,
              compression='gzip')


if __name__ == '__main__':
    Fire(main)

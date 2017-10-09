import bson
from tqdm import tqdm

DATA_DIR = "../../data"

if __name__ == '__main__':
    test = bson.decode_file_iter(open(f'{DATA_DIR}/test.bson', 'rb'))
    with open(f'{DATA_DIR}/test.csv', 'w') as f:
        f.write('product_id,images_count\n')
        for item in tqdm(test, total=1768182):
            product_id = item['_id']
            imgs_count = len(item['imgs'])
            f.write(f'{product_id},{imgs_count}\n')
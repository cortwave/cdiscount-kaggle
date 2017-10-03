import bson
from tqdm import tqdm

DATA_DIR = "../../data"

if __name__ == '__main__':
    train = bson.decode_file_iter(open(f'{DATA_DIR}/train.bson', 'rb'))
    with open(f'{DATA_DIR}/train.csv', 'w') as f:
        f.write('product_id,category_id,image_id\n')
        ix = 0
        for item in tqdm(train, total=7069896):
            product_id = item['_id']
            category_id = item['category_id']
            imgs_count = len(item['imgs'])
            for i in range(imgs_count):
                f.write(f'{product_id},{category_id},{product_id}_{i}\n')
                ix += 1

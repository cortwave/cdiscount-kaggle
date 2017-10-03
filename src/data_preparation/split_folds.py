import pandas as pd
from fire import Fire
from tqdm import tqdm

class Splitter(object):
    """Class to split train csv on folds"""

    def split(self, n_folds=5, train_csv="../../data/train.csv", output_dir="../../data"):
        print(f"n_folds = {n_folds}")
        print(f"train_csv = {train_csv}")
        self._prepare_folds(n_folds, output_dir)
        train = pd.read_csv(train_csv)
        train_grouped = train.groupby(['category_id'])
        categories = train['category_id'].unique()
        for category in tqdm(categories):
            category_group = train_grouped.get_group(category)
            products_grouped = category_group.groupby(['product_id'])
            products_id = category_group['product_id'].unique()
            for ix, product_id in tqdm(enumerate(products_id)):
                images = products_grouped.get_group(product_id)
                n_fold = ix % n_folds
                for image in images.iterrows():
                    img = image[1]
                    self._add_to_fold(n_fold, img['image_id'], img['product_id'], img['category_id'])
        self._close_files()

    def _prepare_folds(self, n_folds, output_dir):
        self.files = []
        for i in range(n_folds):
            with open(f"{output_dir}/fold_{i}.csv", "w") as f:
                f.write("image_id,category_id,product_id\n")
            self.files.append(open(f"{output_dir}/fold_{i}.csv", "a"))

    def _add_to_fold(self, n_fold, image_id, product_id, category_id):
            self.files[n_fold].write(f"{image_id},{product_id},{category_id}\n")

    def _close_files(self):
        for f in self.files:
            f.close()


if __name__ == '__main__':
    Fire(Splitter)
import pandas as pd
import numpy as np


def main():
    train_df = pd.read_csv("../../data/train.csv")
    categories = np.sort(train_df['category_id'].unique())
    new_labels = np.arange(0, len(categories))
    labels_df = pd.DataFrame(columns=["category_id", "label_id"])
    labels_df["category_id"] = categories
    labels_df["label_id"] = new_labels
    cat_labs = {}
    for c, l in zip(categories, new_labels):
        cat_labs[c] = l
    labels_df.to_csv("../../data/labels_map.csv", index=None)
    train_df['label'] = train_df['category_id'].map(lambda x: cat_labs[x])
    train_df.to_csv("../../data/train.csv", index=None)


if __name__ == '__main__':
    main()
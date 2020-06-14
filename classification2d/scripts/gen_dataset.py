import pandas as pd


def main(train_csv, train_label_csv):
    train_df = pd.read_csv(train_csv)
    train_name, train_annotation = train_df['image_name'], train_df['annotation']
    train_label = []
    for idx in range(len(train_name)):
        annotation = train_annotation[idx]
        if isinstance(annotation, float) or len(annotation) == 0:
            train_label.append([train_name[idx], 0, train_annotation[idx]])
        else:
            train_label.append([train_name[idx], 1, train_annotation[idx]])

    train_label_df = pd.DataFrame(data=train_label, columns=['image_name', 'label', 'annotation'])
    train_label_df.to_csv(train_label_csv, index=False)


if __name__ == '__main__':

    in_train_csv = '/mnt/projects/CXR_Object/dev.csv'
    out_train_csv = '/mnt/projects/CXR_Object/dev_label.csv'
    main(in_train_csv, out_train_csv)
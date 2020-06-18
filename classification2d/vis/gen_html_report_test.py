import argparse
import pandas as pd

from classification2d.vis.gen_html_report import gen_html_report


def parse_and_check_arguments():
    """
    Parse input arguments and raise error if invalid.
    """
    default_image_folder = '/mnt/projects/CXR_Object/data/dev'
    default_label_file = '/mnt/projects/CXR_Object/dataset/dev_label.csv'
    default_prediction_file = '/mnt/projects/CXR_Object/results/model_0617_2020/dev_prediction.csv'
    # default_resolution = [1.5, 1.5, 1.5]
    # default_contrast_range = None
    default_output_folder = '/mnt/projects/CXR_Object/results/vis/classification'
    default_generate_pictures = False

    parser = argparse.ArgumentParser(
        description='Snapshot three planes centered around landmarks.')
    parser.add_argument('--image-folder', type=str,
                        default=default_image_folder,
                        help='Folder containing the source data.')
    parser.add_argument('--label-file', type=str,
                        default=default_label_file,
                        help='')
    parser.add_argument('--prediction-file', type=str,
                        default=default_prediction_file,
                        help='')
    # parser.add_argument('--resolution', type=list,
    #                     default=default_resolution,
    #                     help="Resolution of the snap shot images.")
    # parser.add_argument('--contrast_range', type=list,
    #                     default=default_contrast_range,
    #                     help='Minimal and maximal value of contrast intensity window.')
    parser.add_argument('--output_folder', type=str,
                        default=default_output_folder,
                        help='Folder containing the generated html report.')
    parser.add_argument('--generate_pictures', type=bool,
                        default=default_generate_pictures,
                        help='Whether generating the pictures for the html report.')

    return parser.parse_args()


def main():

    args = parse_and_check_arguments()

    labels_df = pd.read_csv(args.label_file, na_filter=False)
    print(f'{len(labels_df)} pictures in the label files')

    preds_df = pd.read_csv(args.prediction_file, na_filter=False)
    print(f'{len(preds_df)} pictures in the prediction files')

    image_names, image_labels = labels_df['image_name'], labels_df['label']
    image_label_dict = dict(zip(image_names, image_labels))

    image_names, image_predictions = preds_df['image_name'], preds_df['prediction']
    image_prediction_dict = dict(zip(image_names, image_predictions))

    usage_flag = 1
    gen_html_report([image_label_dict, image_prediction_dict], usage_flag, args.output_folder)


if __name__ == '__main__':

    main()
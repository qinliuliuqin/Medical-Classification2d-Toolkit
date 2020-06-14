import argparse

from classification2d.core.cls_train import train


def main():

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    long_description = "Training engine for 2d medical image classification."
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default='/home/ql/projects/Medical-Classification2d-Toolkit/classification2d/config/train_config.py',
                        help='configure file for 2d medical image classification training.')
    args = parser.parse_args()
    train(args.input)


if __name__ == '__main__':
    main()

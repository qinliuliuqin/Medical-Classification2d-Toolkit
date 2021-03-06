import importlib
import numpy as np
import os
import shutil
from sklearn.metrics import roc_auc_score
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from classification2d.dataset.sampler import EpochConcateSampler
from classification2d.dataset.dataset import ClassificationDataset
from classification2d.loss.focal_loss import FocalLoss
from classification2d.utils.file_io import load_config, setup_logger
from classification2d.utils.model_io import load_checkpoint, save_checkpoint
from classification2d.network.module.weight_init import kaiming_weight_init


def train(train_config_file):
    """ Medical image segmentation training engine
    :param train_config_file: the input configuration file
    :return: None
    """
    assert os.path.isfile(train_config_file), 'Config not found: {}'.format(train_config_file)

    # load config file
    train_cfg = load_config(train_config_file)

    # clean the existing folder if training from scratch
    model_folder = os.path.join(train_cfg.general.model_save_dir)
    if os.path.isdir(model_folder):
        if train_cfg.general.resume_epoch < 0:
            shutil.rmtree(model_folder)
            os.makedirs(model_folder)
    else:
        os.makedirs(model_folder)

    # copy training and inference config files to the model folder
    shutil.copy(train_config_file, os.path.join(model_folder, 'train_config.py'))
    infer_config_file = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'infer_config.py'))
    shutil.copy(infer_config_file, os.path.join(train_cfg.general.model_save_dir, 'infer_config.py'))

    # enable logging
    log_file = os.path.join(model_folder, 'train_log.txt')
    logger = setup_logger(log_file, 'cls2d')

    # control randomness during training
    np.random.seed(train_cfg.general.seed)
    torch.manual_seed(train_cfg.general.seed)
    if train_cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(train_cfg.general.seed)

    # load training dataset
    train_dataset = ClassificationDataset(
        mode='train',
        data_folder=train_cfg.general.train_data_folder,
        label_file=train_cfg.general.train_label_file,
        transforms=train_cfg.dataset.train_transforms
    )
    train_data_loader = DataLoader(
        train_dataset,
        sampler=EpochConcateSampler(train_dataset, train_cfg.train.epochs, train_cfg.general.seed),
        batch_size=train_cfg.train.batchsize,
        num_workers=train_cfg.train.num_threads,
        pin_memory=True
    )

    # load validation dataset
    val_dataset = ClassificationDataset(
        mode='val',
        data_folder=train_cfg.general.val_data_folder,
        label_file=train_cfg.general.val_label_file,
        transforms=train_cfg.dataset.val_transforms
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    if train_cfg.net.name.find('resnet') == 0:
        model_module = importlib.import_module('classification2d.network.resnet')
        model = model_module.get_classification_model(
            classifier_name=train_cfg.net.name,
            num_classes=train_cfg.dataset.num_classes,
            pretrained=train_cfg.net.pre_trained)

    else:
        raise ValueError('Unsupported module type.')

    kaiming_weight_init(model)

    if train_cfg.general.num_gpus > 0:
        model = nn.parallel.DataParallel(model, device_ids=list(range(train_cfg.general.num_gpus)))
        model = model.cuda()

    # training optimizer
    optim = train_cfg.train.optimizer
    if optim.name == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=train_cfg.train.lr, momentum=optim.sgd_momentum, weight_decay=optim.weight_decay
        )

    elif optim.name == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg.train.lr, betas=optim.adam_betas
        )

    else:
        raise ValueError('Unsupported optimizer type!')

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim.step_size, gamma=optim.gamma)

    if train_cfg.loss.name == 'Focal':
        # reuse focal loss if exists
        loss_func = FocalLoss(
            class_num=train_cfg.dataset.num_classes,
            alpha=train_cfg.loss.obj_weight,
            gamma=train_cfg.loss.focal_gamma,
            use_gpu=train_cfg.general.num_gpus > 0
        )
    else:
        raise ValueError('Unknown loss function')

    # load checkpoint if resume epoch > 0
    if train_cfg.general.resume_epoch >= 0:
        last_save_epoch, last_save_batch = load_checkpoint(train_cfg.general.resume_epoch, model, optimizer, model_folder)
    else:
        last_save_epoch, last_save_batch = 0, 0

    writer = SummaryWriter(os.path.join(model_folder, 'tensorboard'))

    # loop over batches
    batch_idx, epoch_idx, last_epoch, max_auc = 0, 0, 0, 0
    for images, labels in train_data_loader:
        begin_t = time.time()

        if train_cfg.general.num_gpus > 0:
            images, labels = images.cuda(), labels.cuda()

        # clear previous gradients
        optimizer.zero_grad()

        # network forward and backward
        outputs = nn.Softmax(1)(model(images))
        train_loss = loss_func(outputs, labels)
        train_loss.backward()

        # update weights
        optimizer.step()

        epoch_idx = batch_idx * train_cfg.train.batchsize // len(train_dataset)
        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / train_cfg.train.batchsize

        epoch_updated = epoch_idx > last_epoch
        if epoch_updated:
            last_epoch = epoch_idx
            lr_scheduler.step()

        # print training loss per batch
        if batch_idx % train_cfg.train.print_freq == 0:
            msg = 'epoch: {}, batch: {}, lr: {:.6f}, train_loss: {:.4f}, time: {:.4f} s/vol'
            msg = msg.format(
                epoch_idx + last_save_epoch,
                batch_idx + last_save_batch,
                optimizer.param_groups[0]["lr"],
                train_loss.item(),
                sample_duration
            )
            logger.info(msg)

        # validation, only used for binary classification
        if epoch_idx != 0 and (epoch_idx % train_cfg.train.save_epochs == 0) and epoch_updated:
            model.eval()

            with torch.no_grad():
                val_pred_probs, val_pred_labels, val_labels = [], [], []
                for image, label in val_data_loader:
                    if train_cfg.general.num_gpus > 0:
                        image, label = image.cuda(), label.cuda()

                    val_labels.append(label.cpu())
                    val_pred_prob = nn.Softmax(1)(model(image))
                    val_pred_probs.append(val_pred_prob[0][1].detach().cpu())

                    _, val_pred_label = val_pred_prob.max(1)
                    val_pred_labels.append(val_pred_label.cpu())

            number = 0
            for i in range(len(val_pred_labels)):
                if val_pred_labels[i] == val_labels[i]:
                    number += 1

            acc = number / len(val_labels)
            auc = roc_auc_score(val_labels, val_pred_probs)

            msg = 'epoch: {} | val acc: {:.4f} | val auc: {:.4f}'.format(epoch_idx + last_save_epoch, acc, auc)
            logger.info(msg)

            if auc > max_auc:
                max_auc = auc
                msg = 'best epoch: {} | val acc: {:.4f} | best val auc: {:.4f}'.format(epoch_idx + last_epoch, acc, auc)
                logger.info(msg)
                save_checkpoint(model, optimizer, epoch_idx + last_save_epoch, batch_idx + last_save_batch, train_cfg)

            model.train()

        writer.add_scalar('Train/Loss', train_loss.item(), batch_idx)

    writer.close()

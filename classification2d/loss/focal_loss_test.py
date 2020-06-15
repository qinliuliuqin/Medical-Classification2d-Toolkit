import torch

from segmentation2d.loss.focal_loss import FocalLoss


def test_focal_loss():
    batch_size, num_class, dim_y, dim_x = 16, 2, 64, 64
    in_tensor = torch.rand([batch_size, num_class, dim_y, dim_x])
    in_target = torch.rand([batch_size, 1, dim_y, dim_x])
    in_target = (in_target > 0.5).float() * 1

    loss_func = FocalLoss(class_num=num_class, alpha=[1] * num_class, gamma=2.0, use_gpu=False)
    loss = loss_func(in_tensor, in_target)

    print(loss)

if __name__ == '__main__':

    test_focal_loss()
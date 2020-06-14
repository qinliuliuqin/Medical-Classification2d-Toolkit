import torch
import unittest

from classification2d.network.resnet import get_classification_model


class test_inception3(unittest.TestCase):

  def setUp(self):
    self.kMega = 1e6

  def test_resnet18_model_parameters(self):
    num_classes = 2
    model = get_classification_model('resnet18', num_classes, False)
    if torch.cuda.is_available():
      model = model.cuda()
    model_params = (sum(p.numel() for p in model.parameters()) / self.kMega)

    expected_model_params = 11.177538  # mb
    self.assertLess(abs(model_params - expected_model_params), 1e-6)

  def test_resnet18_output_channels(self):
    batch_size, num_classes, in_channels = 1, 2, 3
    (dim_x, dim_y) = (512, 512)

    model = get_classification_model('resnet18', num_classes, False)
    if torch.cuda.is_available():
      model = model.cuda()

    in_images = torch.zeros([batch_size, in_channels, dim_y, dim_x])
    if torch.cuda.is_available():
      in_images = in_images.cuda()

    outputs = model(in_images)

    self.assertEqual(outputs.size()[0], batch_size)
    self.assertEqual(outputs.size()[1], num_classes)


if __name__ == '__main__':
  unittest.main()
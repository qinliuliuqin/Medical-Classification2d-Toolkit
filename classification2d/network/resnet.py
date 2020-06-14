from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def get_classification_model(classifier_name, num_classes, pretrained):
    """
    Get the detection model
    :param num_classes:
    :return:
    """
    if classifier_name == 'resnet18':
        model = resnet18(pretrained, num_classes=num_classes)

    elif classifier_name == 'resnet34':
        model = resnet34(pretrained, num_classes=num_classes)

    elif classifier_name == 'resnet50':
        model = resnet50(pretrained, num_classes=num_classes)

    elif classifier_name == 'resnet101':
        model = resnet101(pretrained, num_classes=num_classes)

    elif classifier_name == 'resnet152':
        model = resnet152(pretrained, num_classes=num_classes)

    else:
        raise ValueError('Unsupported resnet type.')

    return model

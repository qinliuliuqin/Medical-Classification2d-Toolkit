from easydict import EasyDict as edict
import torchvision.transforms as transforms


__C = edict()
cfg = __C


__C.data_transforms = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

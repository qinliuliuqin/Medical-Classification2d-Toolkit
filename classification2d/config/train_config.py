from easydict import EasyDict as edict
import torchvision.transforms as transforms


__C = edict()
cfg = __C


##################################
# general parameters
##################################

__C.general = {}

# image-segmentation pair list for training
__C.general.train_label_file = '/shenlab/lab_stor6/projects/PIC_TNSCUI2020/datasets/train.csv'

__C.general.train_data_folder = '/shenlab/lab_stor6/projects/PIC_TNSCUI2020/datasets/train.csv'

# image-segmentation pair list for validation
__C.general.val_label_file = '/shenlab/lab_stor6/projects/PIC_TNSCUI2020/datasets/test.csv'

__C.general.val_data_folder = '/shenlab/lab_stor6/projects/PIC_TNSCUI2020/datasets/train.csv'

# the output of training models and logs
__C.general.model_save_dir = '/shenlab/lab_stor6/qinliu/PIC_TNSCUI2020/models/model_0601_2020'

# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = -1

# the number of GPUs used in training. Set to 0 if using cpu only.
__C.general.num_gpus = 1

# random seed used in training (debugging purpose)
__C.general.seed = 0


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of classes
__C.dataset.num_classes = 2

# transform for training set
__C.dataset.train_transforms = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# transform for validation set
__C.dataset.val_transforms = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

##################################
# training loss
##################################

__C.loss = {}

# the name of loss function to use
# Focal: Focal loss, supports binary-class and multi-class segmentation
# Dice: Dice Similarity Loss which supports binary and multi-class segmentation
__C.loss.name = 'Focal'

# the weight for each class including background class
# weights will be normalized
__C.loss.obj_weight = [1/2, 1/2]

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2


##################################
# net
##################################

__C.net = {}

# the network name
__C.net.name = 'vdnet2d'

##################################
# training parameters
##################################

__C.train = {}

# the number of training epochs
__C.train.epochs = 1001

# the number of samples in a batch
__C.train.batchsize = 6

# the number of threads for IO
__C.train.num_threads = 6

# the learning rate
__C.train.lr = 1e-4

# the beta in Adam optimizer
__C.train.betas = (0.9, 0.999)

# the number of batches to save model
__C.train.save_epochs = 100


###################################
# debug parameters
###################################

__C.debug = {}

# whether to save input crops
__C.debug.save_inputs = True

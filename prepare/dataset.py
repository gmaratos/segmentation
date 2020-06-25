import os
import torch
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    """VOC Segmentation dataset object, written by hand because I think the
    transforms are messed up so I am making my own"""

    def __init__(self, root: str, transforms=None):

        image_path = os.path.join(root, 'VOC2012', 'JPEGImages')
        target_path = os.path.join(root, 'VOC2012', 'SegmentationClass')
        #extract files names from target because jpegimages has many unlabled images
        file_names = [line.split('.')[0].strip() for line in os.listdir(target_path)]
        #future optimization could be to build static name list input/target
        #instead of constructing the paths on the fly in __getitem__
        self.image_path = image_path
        self.target_path = target_path
        self.file_names = file_names
        self.transforms = transforms

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):

        image_location = os.path.join(self.image_path, self.file_names[index] + '.jpg')
        target_location = os.path.join(self.target_path, self.file_names[index] + '.png')

        img = Image.open(image_location).convert('RGB')
        target = Image.open(target_location)

        import pdb;pdb.set_trace()




































































































#def get_transform(train):
#    """build transform to pytorch reference specifications, if training then
#    do some additional augmentation. The RandomResize function seems to be
#    missing from the documentation, also the 'replacement' RandomResizedCrop
#    does not seem to do exactly what is happening here."""
#
#    base_size = 520
#    crop_size = 480
#
#    min_size = int((0.5 if train else 1.0) * base_size)
#    max_size = int((2.0 if train else 1.0) * base_size)
#    transforms = []
#    transforms.append(T.RandomResize(min_size, max_size))
#    if train:
#        transforms.append(T.RandomHorizontalFlip(0.5))
#        transforms.append(T.RandomCrop(crop_size))
#    transforms.append(T.ToTensor())
#    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]))
#
#    return T.Compose(transforms)
#
#def cat_list(images, fill_value=0):
#    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
#    batch_shape = (len(images),) + max_size
#    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
#    for img, pad_img in zip(images, batched_imgs):
#        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
#    return batched_imgs
#
#def collate_fn(batch):
#    images, targets = list(zip(*batch))
#    batched_imgs = cat_list(images, fill_value=0)
#    batched_targets = cat_list(targets, fill_value=255)
#    return batched_imgs, batched_targets
#
#def build_dataloader(dataset: str):
#    """
#    currently supported:
#    VOC
#    """
#
#    if dataset == 'VOC':
#        validation_dataset = torchvision.datasets.VOCSegmentation(
#            'VOC', image_set='val', transform=get_trainsform(False)
#        )
#

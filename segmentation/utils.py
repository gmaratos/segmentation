import os
import torch

def create_dir(name: str):
    if not os.path.exists(name):
        os.makedirs(name)

def record_pil_image(img):
    create_dir('sample_images')

    name = f"image_{len(os.listdir('sample_images'))}.png"
    img.save(os.path.join('sample_images', name), 'PNG')

#functions for combining the VOC batches
def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

def build_dataloaders(train_dataset, train_sampler, test_dataset, test_sampler, batch_size, num_workers=0):
    #create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers,
        collate_fn=collate_fn, drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        sampler=test_sampler, num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_dataloader, test_dataloader

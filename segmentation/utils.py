import os
import torch

__all__ = ['ConfusionMatrix']

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

class ConfusionMatrix:
    """this is just ripped from the pytorch reference"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, output):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n) #prune class 255
            #this is a way to flatten the address space for bincount
            inds = n * target[k].to(torch.int64) + output[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        """used in parallel processing"""
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

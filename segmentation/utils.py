import os
import tqdm
import torch
import torch.distributed as dist

__all__ = ['ConfusionMatrix']

def create_dir(name: str):
    """create a directory if it does not already exist"""
    if not os.path.exists(name):
        os.makedirs(name)

def record_pil_image(img):
    """save images to the sample directory. I used this mostly for testing
    image augmentations"""
    create_dir('sample_images')

    name = f"image_{len(os.listdir('sample_images'))}.png"
    img.save(os.path.join('sample_images', name), 'PNG')

#functions taken from pytorch for building a batch from the dataset
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
#end of pytorch functions

def build_dataloaders(train_dataset, train_sampler, test_dataset, test_sampler, batch_size, num_workers=0):
    """constructs the dataloader objects from the datasets and samplers"""
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
    """used in evaluation, when calculating IoU"""
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

#taken from pytorch for distributed processing
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    #grab environment variables if using torch distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not in distributed mode')
        args.distributed = False
        return
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)

    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    setup_for_distributed(args.rank == 0)
#end of distributed

def save_on_master(*args, **kwargs):
    """save all information about the model that is relevant, taken from pytorch"""
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            torch.save(*args, **kwargs)
    else:
        torch.save(**args, **kwargs)

def forward_pass(model, dataloader, criterion, optimizer, lr_scheduler, device, train=False):
    """ forward pass infers the number of classes, for evaluation, from the dataset
    object. Run a single forward pass over the data """

    #training pass
    if train:
        model.train()
        t, losses = tqdm.tqdm(dataloader, leave=True), []
        for (x, y) in t:
            t.set_description(desc='Train')
            #forward pass on batch
            x, y = x.to(device), y.to(device)
            prediction = model(x)['out']
            loss = criterion(prediction, y)

            #update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            #loss.item() is called twice!
            t.set_postfix(loss=f'{loss.item():.3e}')
            losses.append(loss.item())
        return sum(losses)/len(losses)
    else:
        #evaluation
        num_classes = dataloader.dataset.num_classes
        confmat = ConfusionMatrix(num_classes)
        model.eval()
        t = tqdm.tqdm(dataloader, leave=True)
        with torch.no_grad():
            for (x, y) in t:
                t.set_description(desc='Test')
                #forward pass on batch
                x, y = x.to(device), y.to(device)
                prediction = model(x)['out']

                confmat.update(y.flatten(), prediction.argmax(1).flatten())
            confmat.reduce_from_all_processes()

        return confmat

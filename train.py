import os
import tqdm
import torch
import torchvision
import segmentation

import torch.nn.functional as F

def criterion_fn(inputs, target):
    return F.cross_entropy(inputs, target, ignore_index=255)

def fit(args, num_workers=0):

    #extract some args
    device = args.device

    epochs = args.epochs
    batch_size = args.batch_size

    dataset_name = args.dataset
    model_name = args.model

    resume = args.resume

    distributed = args.distributed

    #create results dir
    segmentation.utils.create_dir('checkpoints')
    root_path = os.path.join('checkpoints', model_name)
    segmentation.utils.create_dir(root_path)

    #construct dataset objects
    dataset = segmentation.datasets.__dict__[dataset_name]
    train_dataset = dataset(dataset_name, 'train')
    test_dataset = dataset(dataset_name, 'val')

    #build sampler this will change if we are doing parallel processing
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    #create data loaders
    train_dataloader, test_dataloader = segmentation.utils.build_dataloaders(
        train_dataset, train_sampler, test_dataset, test_sampler,
        batch_size,
    )

    #create model, also initialize some things for parallelism if used
    #model = torchvision.models.segmentation.__dict__[model_name](
    #    num_classes=train_dataset.num_classes,
    #)
    model = segmentation.models.__dict__[model_name](
        num_classes=train_dataset.num_classes,
    )
    model.to(device)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_no_ddp = model

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_no_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_no_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_no_ddp.classifier.parameters() if p.requires_grad]},
    ]

    #create optimizer
    epochs = 100
    optimizer = torch.optim.SGD(
	params_to_optimize,
	lr=1e-2, momentum=0.9, weight_decay=0,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (len(train_dataloader)*epochs) ** 0.9)
    )

    #resuming model
    start_epoch = 1
    if resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, epochs+1):
        print(f'Start of Epoch {epoch}')
        #train pass
        loss = segmentation.utils.forward_pass(
            model, train_dataloader,
            criterion_fn, optimizer, lr_scheduler,
            device, train = True
        )
        print(f'\tFinal loss: {loss:.3e}')
        #evaluation
        result = segmentation.utils.forward_pass(
            model, train_dataloader,
            criterion_fn, optimizer, lr_scheduler,
            device, train = False
        )
        print(result)

        #checkpoint
        segmentation.utils.save_on_master({
            'model': model_no_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
            'model_name': model_name,
        }, os.path.join(root_path, f'{epoch}.pth'))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Segmentation')

    parser.add_argument('--dataset', default='VOC', help='dataset')
    parser.add_argument('--model', default='VGG16_FCN', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int, help='epoch number')
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--word-size', default=1, type=int, help='#of processes')
    parser.add_argument('--dist-url', default='env://', help='url for distributed mode')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    segmentation.utils.init_distributed_mode(args)
    fit(args)

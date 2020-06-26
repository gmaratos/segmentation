import os
import torch
import torchvision
import segmentation

def fit(model_name: str, dataset_name: str, batch_size: int, device, num_workers=0):

    #create results dir
    segmentation.utils.create_dir('train')
    results_root = 'train'

    #construct dataset objects
    dataset = segmentation.datasets.__dict__[dataset_name]
    train_dataset = dataset(dataset_name, 'train')
    test_dataset = dataset(dataset_name, 'val')

    #build sampler this will change if we are doing parallel processing
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    #create data loaders
    train_dataloader, test_dataloader = segmentation.utils.build_dataloaders(
        train_dataset, train_sampler, test_dataset, test_sampler,
        batch_size,
    )

    #create model
    model = torchvision.models.segmentation.__dict__[model_name](
        num_classes=train_dataset.num_classes,
    )

    #will make more sense when I add parallelism
    model_no_ddp = model

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
    import pdb;pdb.set_trace()

fit('fcn_resnet50', 'VOC', 10, None)

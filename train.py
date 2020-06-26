import os
import tqdm
import torch
import torchvision
import segmentation

import torch.nn.functional as F

def criterion_fn(inputs, target):
    return F.cross_entropy(inputs, target, ignore_index=255)

def forward_pass(model, dataloader, criterion, optimizer, lr_scheduler, device, train=False):
    """ forward pass infers the number of classes, for evaluation, from the dataset
    object """

    #training pass
    if train:
        model.train()
        t, losses = tqdm.tqdm(dataloader), []
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
        t.set_postfix(loss=f'{sum(losses)/len(losses):.3e}')
        return
    else:
        #evaluation
        num_classes = dataloader.dataset.num_classes
        confmat = segmentation.utils.ConfusionMatrix(num_classes)
        model.eval()
        t = tqdm.tqdm(dataloader)
        with torch.no_grad():
            for (x, y) in t:
                t.set_description(desc='Test')
                #forward pass on batch
                x, y = x.to(device), y.to(device)
                prediction = model(x)['out']

                confmat.update(y.flatten(), prediction.argmax(1).flatten())
            confmat.reduce_from_all_processes()

        return confmat

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
    model.to(device)

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

    for epoch in range(1, epochs+1):
        forward_pass(
            model, train_dataloader,
            criterion_fn, optimizer, lr_scheduler,
            device, train = True
        )
        result = forward_pass(
            model, train_dataloader,
            criterion_fn, optimizer, lr_scheduler,
            device, train = False
        )
        print(result)

device = torch.device('cuda:0')
fit('fcn_resnet50', 'VOC', 2, device)

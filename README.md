This is a library for running semantic segmentation experiments. I used the code in the pytorch vision reference repository as a guide. It is capable of using either a single gpu or multiple, with the torch distributed package.

## How to Run:
Example of how to train a model in a single thread
```
python train.py --batch-size 10 --dataset VOC --model VGG16_FCN
```

Example of how to train using 3 graphics cards on one machine
```
python -m torch.distributed.launch --nproc_per_node=3 --use_env train.py --batch-size 10 --dataset VOC --model VGG16_FCN
```

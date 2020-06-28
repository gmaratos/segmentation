This is a library for running semantic segmentation experiments. I used the code in the pytorch vision reference repository as a guide. It is capable of using either a single gpu or multiple, with the torch distributed package. For the training I used [the pytorch refrence code](https://github.com/pytorch/vision/tree/master/references/segmentation) and I also took lots of inspiration from [the vision library](https://github.com/pytorch/vision) in general. This is, by no means, all my ideas but instead a rework to simplify things so that I can understand them and use them in my future projects.

## How to Run:
Example of how to train a model in a single thread
```
python train.py --batch-size 10 --dataset VOC --model VGG16_FCN
```

Example of how to train using 3 graphics cards on one machine
```
python -m torch.distributed.launch --nproc_per_node=3 --use_env train.py --batch-size 10 --dataset VOC --model VGG16_FCN
```

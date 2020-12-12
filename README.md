# End-to-end lyrics alignment implementation

Implementation of paper "End-to-end lyrics alignment for polyphonic music using an audio-to-character recognition model" ([link](https://ieeexplore.ieee.org/iel7/8671773/8682151/08683470.pdf?casa_token=uETIsIkBb8kAAAAA:iUaY865SXUM1jmNauTmJTsKyRIf-Yvd4PtBIf0ll60lzXbVf2K2cIoGI_Rqi9NVoSdYGE7F58Q)) based on the pytorch implementation of [Wave-U-Net](https://github.com/f90/Wave-U-Net-Pytorch).

The input and output size are reduced to stablize training on the [DALI dataset](https://github.com/gabolsgabs/DALI).

To train the model, pull the DALI wrapper somewhere and link it from the root of this repository.

```
    ln -s path/to/dali_wrapper/ DALI
```

Then run the following command. Set `--cuda` flag if you have GPUs.

```
python train.py --dataset_dir path/to/DALI_v2.0/
```

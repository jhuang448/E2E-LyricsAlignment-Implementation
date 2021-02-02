import argparse
import os

import time
from functools import partial

import torch
import pickle, sys
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim import Adam
from tqdm import tqdm

import utils
# from data import get_musdb_folds, SeparationDataset, random_amplify, crop
from data import get_dali_folds, LyricsAlignDataset
from test import predict, validate
from waveunet import WaveunetLyrics

utils.seed_torch(2742)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def main(args):
    # torch.backends.cudnn.benchmark=True # This makes dilated conv much faster for CuDNN 7.5

    # MODEL
    down_features = [args.features*i for i in range(1, args.down_levels+2)] # [args.features*2**i for i in range(0, args.down_levels+1)]
    up_features = down_features[-args.up_levels:]

    model = WaveunetLyrics(num_inputs=args.channels, num_channels=[down_features, up_features], num_outputs=args.num_class,
                           kernel_size=[15, 5], input_sample=250000, output_sample=123904, depth=args.depth,
                           strides=args.strides, conv_type=args.conv_type, res=args.res)

    target_frame = int(123904/1024)

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    if 'cuda' in device:
        model = utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    # print('model: ', model)
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    import datetime
    current = datetime.datetime.now()
    writer = SummaryWriter(args.log_dir + current.strftime("%m:%d:%H:%M"))

    ### DATASET
    # dali_split = get_dali_folds(args.dataset_dir, level="words", sepa_audio_path=args.sepa_dir)
    dali_split = {"train": [], "val": []} # h5 files already saved

    # If not data augmentation, at least crop targets to fit model output shape
    # crop_func = partial(crop, shapes=model.shapes)

    val_data = LyricsAlignDataset(dali_split, "val", args.sr, model.shapes, args.hdf_dir, sepa=False, dummy=args.dummy)
    train_data = LyricsAlignDataset(dali_split, "train", args.sr, model.shapes, args.hdf_dir, sepa=True, dummy=args.dummy, aug=True)

    print("dummy?", args.dummy, len(train_data), len(val_data))

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                             worker_init_fn=utils.worker_init_fn, sampler=RandomSampler(data_source=train_data),
                                             collate_fn=utils.my_collate)

    ##### TRAINING ####

    # Set up the loss function
    if args.loss == "CTC":
        criterion = nn.CTCLoss(blank=28, zero_infinity=True).to(device)
    else:
        raise NotImplementedError("Couldn't find this loss!")

    # Set up optimiser
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf}

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = utils.load_model(model, optimizer, args.load_model, args.cuda)

    print('TRAINING START')
    while state["worse_epochs"] < args.patience:
        print("Training one epoch from epoch " + str(state["epochs"]))
        avg_time = 0.
        model.train()
        train_loss = 0.
        with tqdm(total=len(train_data) // args.batch_size) as pbar:

            for example_num, _data in enumerate(dataloader):

                x,  targets, seqs = _data

                x = utils.move_data_to_device(x, device)
                seqs = [utils.move_data_to_device(seq, device) for seq in seqs]

                t = time.time()

                # Set LR for this iteration
                # utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                # utils.update_lr(optimizer, state["epochs"], 1, args.lr)

                writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])
                # print(utils.get_lr(optimizer))

                # Compute loss for each instrument/model
                optimizer.zero_grad()
                model.zero_grad()

                avg_loss = utils.compute_loss(model, x, seqs, criterion, compute_grad=True)

                optimizer.step()

                state["step"] += 1

                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                writer.add_scalar("train/step_loss", avg_loss, state["step"])

                # print(avg_loss)
                train_loss += avg_loss

                pbar.set_description("Current loss: {:.4f}".format(avg_loss))
                pbar.update(1)

                if example_num == len(train_data) // args.batch_size:
                    break

        train_loss /= (len(train_data) // args.batch_size)
        print("Train Loss: {:.4f}".format(train_loss))
        writer.add_scalar("train/epoch_loss", train_loss, state["epochs"])

        # VALIDATE
        val_loss = validate(args, model, target_frame, criterion, val_data, device)
        val_loss /= (len(val_data) // args.batch_size)
        print("VALIDATION FINISHED: LOSS: " + str(val_loss))
        writer.add_scalar("val/loss", val_loss, state["epochs"])

        # EARLY STOPPING CHECK
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path

        # CHECKPOINT
        print("Saving model...")
        utils.save_model(model, optimizer, state, checkpoint_path)

        state["epochs"] += 1

    writer.close()

if __name__ == '__main__':
    ## TRAIN PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--dummy', action='store_true',
                        help='Use dummy train/val sets (default: False)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--features', type=int, default=24,
                        help='Number of feature channels per layer')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Folder to write logs into')
    parser.add_argument('--dataset_dir', type=str, default="/import/c4dm-datasets/DALI_v2.0/",
                        help='Dataset path')
    parser.add_argument('--sepa', action='store_true',
                        help='Save separated files to hdfs files.')
    parser.add_argument('--sepa_dir', type=str, default="/import/c4dm-datasets/sepa_DALI/audio/",
                        help='Dataset path')
    parser.add_argument('--hdf_dir', type=str, default="/import/c4dm-datasets/sepa_DALI/hdf/",
                        help='HDF5 file path')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Folder to write checkpoints into, e.g. checkpoints/waveunet/')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate in LR cycle (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=5e-5,
                        help='Minimum learning rate in LR cycle (default: 5e-5)')
    parser.add_argument('--cycles', type=int, default=2,
                        help='Number of LR cycles per epoch')
    parser.add_argument('--batch_size', type=int, required=True,
                        help="Batch size, e.g. 16, 32...")
    parser.add_argument('--down_levels', type=int, default=12,
                        help="Number of DS blocks")
    parser.add_argument('--up_levels', type=int, default=2,
                        help="Number of US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=22050,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=1,
                        help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--strides', type=int, default=2,
                        help="Strides in Waveunet")
    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--example_freq', type=int, default=200,
                        help="Write an audio summary into Tensorboard logs every X training iterations")
    parser.add_argument('--loss', type=str, default="CTC",
                        help="CTC")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--num_class', type=int, default=29,
                        help="Number of predicted classes.")
    parser.add_argument('--feature_growth', type=str, default="add",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

    args = parser.parse_args()

    print(args)

    main(args)

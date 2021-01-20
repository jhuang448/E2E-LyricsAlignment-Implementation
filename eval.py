import os, argparse

import torch

from data import JamendoLyricsDataset
from waveunet import WaveunetLyrics
import utils, test

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def main(args):
    # MODEL
    down_features = [args.features*i for i in range(1, args.down_levels+2)] # [args.features*2**i for i in range(0, args.down_levels+1)]
    up_features = down_features[-args.up_levels:]

    model = WaveunetLyrics(num_inputs=args.channels, num_channels=[down_features, up_features],
                           num_outputs=args.num_class,
                           kernel_size=[15, 5], input_sample=250000, output_sample=123904, depth=args.depth,
                           strides=args.strides, conv_type=args.conv_type, res=args.res)

    target_frame = int(123904 / 1024)

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    if 'cuda' in device:
        print("move model to gpu")
        model = utils.DataParallel(model)
        model.cuda()

    # print('model: ', model)
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    print("Loading full model from checkpoint " + str(args.load_model))

    state = utils.load_model(model, None, args.load_model, args.cuda)

    test_data = JamendoLyricsDataset(args.sr, model.shapes, args.hdf_dir, args.dataset, args.audio_dir)

    results = test.predict(args, model, test_data, device)


if __name__ == '__main__':
    ## EVALUATE PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--features', type=int, default=24,
                        help='Number of feature channels per layer')
    parser.add_argument('--audio_dir', type=str, default="/import/c4dm-05/jh008/jamendolyrics/",
                        help='Dataset path')
    parser.add_argument('--dataset', type=str, default="jamendo",
                        help='Dataset name')
    parser.add_argument('--hdf_dir', type=str, default="/import/c4dm-datasets/sepa_DALI/hdf/",
                        help='Dataset path')
    parser.add_argument('--pred_dir', type=str, default="predict_5s",
                        help='prediction path')
    parser.add_argument('--load_model', type=str, default='checkpoints/waveunet_5s/checkpoint_best',
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size")
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
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--num_class', type=int, default=29,
                        help="Number of predicted classes.")
    parser.add_argument('--feature_growth', type=str, default="add",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

    args = parser.parse_args()

    main(args)
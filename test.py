import museval
from tqdm import tqdm

import utils

import numpy as np
import torch, os
import torch.nn as nn
from torch.utils.data.sampler import SequentialSampler
from model_speech import train_audio_transforms

import time
from utils import compute_loss

import matplotlib.pyplot as plt

def predict_old(audio, model):
    if isinstance(audio, torch.Tensor):
        is_cuda = audio.is_cuda()
        audio = audio.detach().cpu().numpy()
        return_mode = "pytorch"
    else:
        return_mode = "numpy"

    expected_outputs = audio.shape[1]

    # Pad input if it is not divisible in length by the frame shift number
    output_shift = model.shapes["output_frames"]
    pad_back = audio.shape[1] % output_shift
    pad_back = 0 if pad_back == 0 else output_shift - pad_back
    if pad_back > 0:
        audio = np.pad(audio, [(0,0), (0, pad_back)], mode="constant", constant_values=0.0)

    target_outputs = audio.shape[1]
    outputs = {key: np.zeros(audio.shape, np.float32) for key in model.instruments}

    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_front_context = model.shapes["output_start_frame"]
    pad_back_context = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    audio = np.pad(audio, [(0,0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)

    # Iterate over mixture magnitudes, fetch network prediction
    with torch.no_grad():
        for target_start_pos in range(0, target_outputs, model.shapes["output_frames"]):

            # Prepare mixture excerpt by selecting time interval
            curr_input = audio[:, target_start_pos:target_start_pos + model.shapes["input_frames"]] # Since audio was front-padded input of [targetpos:targetpos+inputframes] actually predicts [targetpos:targetpos+outputframes] target range

            # Convert to Pytorch tensor for model prediction
            curr_input = torch.from_numpy(curr_input).unsqueeze(0)

            # Predict
            for key, curr_targets in utils.compute_output(model, curr_input).items():
                outputs[key][:,target_start_pos:target_start_pos+model.shapes["output_frames"]] = curr_targets.squeeze(0).cpu().numpy()

    # Crop to expected length (since we padded to handle the frame shift)
    outputs = {key : outputs[key][:,:expected_outputs] for key in outputs.keys()}

    if return_mode == "pytorch":
        outputs = torch.from_numpy(outputs)
        if is_cuda:
            outputs = outputs.cuda()
    return outputs

def predict_song(args, audio_path, model):
    model.eval()

    # Load mixture in original sampling rate
    mix_audio, mix_sr = utils.load(audio_path, sr=None, mono=False)
    mix_channels = mix_audio.shape[0]
    mix_len = mix_audio.shape[1]

    # Adapt mixture channels to required input channels
    if args.channels == 1:
        mix_audio = np.mean(mix_audio, axis=0, keepdims=True)
    else:
        if mix_channels == 1: # Duplicate channels if input is mono but model is stereo
            mix_audio = np.tile(mix_audio, [args.channels, 1])
        else:
            assert(mix_channels == args.channels)

    # resample to model sampling rate
    mix_audio = utils.resample(mix_audio, mix_sr, args.sr)

    sources = predict_old(mix_audio, model)

    # Resample back to mixture sampling rate in case we had model on different sampling rate
    sources = {key : utils.resample(sources[key], args.sr, mix_sr) for key in sources.keys()}

    # In case we had to pad the mixture at the end, or we have a few samples too many due to inconsistent down- and upsamṕling, remove those samples from source prediction now
    for key in sources.keys():
        diff = sources[key].shape[1] - mix_len
        if diff > 0:
            print("WARNING: Cropping " + str(diff) + " samples")
            sources[key] = sources[key][:, :-diff]
        elif diff < 0:
            print("WARNING: Padding output by " + str(diff) + " samples")
            sources[key] = np.pad(sources[key], [(0,0), (0, -diff)], "constant", 0.0)

        # Adapt channels
        if mix_channels > args.channels:
            assert(args.channels == 1)
            # Duplicate mono predictions
            sources[key] = np.tile(sources[key], [mix_channels, 1])
        elif mix_channels < args.channels:
            assert(mix_channels == 1)
            # Reduce model output to mono
            sources[key] = np.mean(sources[key], axis=0, keepdims=True)

        sources[key] = np.asfortranarray(sources[key]) # So librosa does not complain if we want to save it

    return sources

def predict(args, model, test_data, device):

    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)

    if not os.path.exists('pics'):
        os.makedirs('pics')

    # PREPARE DATA
    dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=utils.my_collate)
    model.eval()
    with tqdm(total=len(test_data) // args.batch_size) as pbar, torch.no_grad():
        for example_num, _data in enumerate(dataloader):
            x, idx, meta = _data
            idx = idx[0]
            words, audio_name, audio_length = meta[0]

            x = utils.move_data_to_device(x, device)
            x = x.squeeze(0)

            # Predict
            all_outputs = model(x)

            batch_num, _, output_length = all_outputs.shape

            output_length = all_outputs.shape[2]

            all_outputs = all_outputs.transpose(1,2)
            # print(all_outputs.shape) # batch, length, classes
            _, _, num_classes = all_outputs.shape

            # plt.matshow(np.exp(all_outputs[9]))
            # plt.savefig('./pics/' + audio_name + '_9.png')

            song_pred = all_outputs.data.numpy().reshape(-1, num_classes)
            # print(song_pred.shape) # total_length, num_classes

            resolution = model.shapes["output_frames"] / output_length / args.sr
            total_length = int(audio_length / args.sr // resolution)

            song_pred = song_pred[:total_length, :]
            # print(song_pred.shape)  # total_length, num_classes

            # smoothing
            P_noise = np.random.uniform(low=1e-11, high=1e-10, size=song_pred.shape)
            song_pred = np.log(np.exp(song_pred) + P_noise)

            # Dynamic programming
            word_align, score = utils.alignment(song_pred, words, idx)
            print("\t{}:\t{}".format(audio_name, score))

            # write
            with open(os.path.join(args.pred_dir, audio_name + "_align.csv"), 'w') as f:
                for word in word_align:
                    f.write("{},{}\n".format(word[0] * resolution, word[1] * resolution))

            pbar.update(1)

    return -1

def validate(args, model, target_frame, criterion, val_data, device):
    # PREPARE DATA
    dataloader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             sampler=SequentialSampler(data_source=val_data),
                                             collate_fn=utils.my_collate)

    # VALIDATE
    model.eval()
    total_loss = 0.
    with tqdm(total=len(val_data) // args.batch_size) as pbar, torch.no_grad():
        for example_num, _data in enumerate(dataloader):
            x, _, seqs = _data

            x = utils.move_data_to_device(x, device)
            seqs = [utils.move_data_to_device(seq, device) for seq in seqs]

            avg_loss = utils.compute_loss(model, x, seqs, criterion, compute_grad=False)

            total_loss += avg_loss

            pbar.set_description("Current loss: {:.4f}".format(avg_loss))
            pbar.update(1)

            if example_num == len(val_data) // args.batch_size:
                break

    return total_loss
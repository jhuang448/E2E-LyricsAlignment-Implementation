import os

import soundfile
import torch
import numpy as np
import librosa
import string
import warnings

def compute_output(model, inputs):
    '''
    Computes outputs of model with given inputs. Does NOT allow propagating gradients! See compute_loss for training.
    Procedure depends on whether we have one model for each source or not
    :param model: Model to train with
    :param compute_grad: Whether to compute gradients
    :return: Model outputs, Average loss over batch
    '''
    all_outputs = {}

    if model.separate:
        for inst in model.instruments:
            output = model(inputs, inst)
            all_outputs[inst] = output[inst].detach().clone()
    else:
        all_outputs = model(inputs)

    return all_outputs


def my_collate(batch):
    audio, targets, seqs = zip(*batch)
    audio = np.array(audio)
    targets = list(targets)
    seqs = list(seqs)
    return audio, targets, seqs

def compute_loss(model, inputs, targets, criterion, compute_grad=False):
    '''
    Computes gradients of model with given inputs and targets and loss function.
    Optionally backpropagates to compute gradients for weights.
    Procedure depends on whether we have one model for each source or not
    :param model: Model to train with
    :param inputs: Input mixture
    :param targets: Target sources
    :param criterion: Loss function to use (L1, L2, ..)
    :param compute_grad: Whether to compute gradients
    :return: Model outputs, Average loss over batch
    '''

    loss = 0
    all_outputs = model(inputs)

    batch_num, _, input_length = all_outputs.shape
    # frame_offset = int((input_length - target_frame) / 2)

    # all_outputs = all_outputs[:, :, frame_offset:-frame_offset]
    input_length = all_outputs.shape[2]

    # all_outputs = torch.nn.functional.log_softmax(all_outputs, 1)
    all_outputs = all_outputs.permute(2, 0, 1)
    # print(all_outputs.shape)

    input_lengths = [input_length] * batch_num
    label_lengths = [len(target) for target in targets]

    try:
        loss = criterion(all_outputs, torch.cat(targets), input_lengths, label_lengths)
    except:
        print(all_outputs.shape, type(all_outputs))
        print(targets)
        print(input_lengths, label_lengths)

    if compute_grad:
        loss.backward()

    avg_loss = loss.item()

    return avg_loss

def worker_init_fn(worker_id): # This is apparently needed to ensure workers have different random seeds and draw different examples!
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def load_example(example, sr=22050, mono=True, mode="numpy", offset=0.0, duration=None):
    if example["vocals"] == None or example["accompaniment"] == None:
        # load original
        y, curr_sr = load(example["path"], sr, mono, mode, offset, duration)
    else:
        # load vocals and accompaniment
        y_v, curr_sr = load(example["vocals"], sr, mono, mode, offset, duration)
        y_a, curr_sr = load(example["accompaniment"], sr, mono, mode, offset, duration)

        assert(y_a.shape[1] == y_v.shape[1])
        y = np.concatenate((y_v, y_a), axis=0)

    return y, curr_sr

def load(path, sr=22050, mono=True, mode="numpy", offset=0.0, duration=None):
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    y, curr_sr = librosa.load(path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)

    if len(y.shape) == 1:
        # Expand channel dimension
        y = y[np.newaxis, :]

    if mode == "pytorch":
        y = torch.tensor(y)

    return y, curr_sr

def load_lyrics(lyrics_file):
    from string import ascii_lowercase
    d = {ascii_lowercase[i]: i for i in range(26)}
    d["'"] = 26
    d[" "] = 27
    d["~"] = 28

    # process raw
    with open(lyrics_file + '.raw.txt', 'r') as f:
        raw_lines = f.read().splitlines()
    # concat
    full_lyrics = " ".join(raw_lines)
    # remove multiple spaces
    full_lyrics = " ".join(full_lyrics.split())
    # remove unknown characters
    full_lyrics = "".join([c for c in full_lyrics.lower() if c in d.keys()])
    # full_lyrics = " " + full_lyrics + " "

    # split to words
    with open(lyrics_file + '.words.txt', 'r') as f:
        words_lines = f.read().splitlines()
    idx = []
    last_end = 0
    for i in range(len(words_lines)):
        word = words_lines[i]
        try:
            assert(word[0] in ascii_lowercase)
        except:
            print(word)
        new_word = "".join([c for c in word.lower() if c in d.keys()])
        offset = full_lyrics[last_end:].find(new_word)
        assert (offset >= 0)
        assert(new_word == full_lyrics[last_end+offset:last_end+offset+len(new_word)])
        idx.append([last_end+offset, last_end+offset+len(new_word)])
        last_end += offset + len(new_word)

    return full_lyrics, words_lines, idx

def random_mute(lyrics_len, mute_prob):
    r = np.random.random_sample(size=(lyrics_len,))
    mute = (r > mute_prob)
    return mute

def generate_envelope(audio_len, times_list, mute):
    fade_max = 0.1 * 22050

    env = np.ones(shape=(audio_len,), dtype=np.float)

    last_unmute_end = 0
    i = 0
    while i < len(mute):
        mute_i = mute[i]
        if mute_i == False:
            # first unmuted word after silence
            unmuted_start = np.int(times_list[i, 0])
            env[last_unmute_end:unmuted_start] = 0

            # skip unmuted segs
            while i < len(mute) and mute[i] == False:
                last_unmute_end = np.int(times_list[i, 1])
                i += 1
                continue
        else:
            # skip muted segs
            while i < len(mute) and mute[i] == True:
                i += 1
                continue
    env[last_unmute_end:] = 0

    return env


def mix_vocal_accompaniment(audio, lyrics_list, times_list, mute_prob):

    mute = random_mute(len(lyrics_list), mute_prob)
    env = generate_envelope(audio.shape[1], times_list, mute)

    mix = audio[0, :] * env + audio[1, :]

    lyrics_unmute = np.array(lyrics_list)[~mute]

    return mix, list(lyrics_unmute)

def write_wav(path, audio, sr):
    soundfile.write(path, audio.T, sr, "PCM_16")

def get_lr(optim):
    return optim.param_groups[0]["lr"]

def set_lr(optim, lr):
    for g in optim.param_groups:
        g['lr'] = lr

def update_lr(optimizer, epoch, update_step, lr):
    new_lr = lr / (((epoch // (update_step * 3)) * 2) + 1)
    new_lr = np.max([1e-5, new_lr])


    set_lr(optimizer, new_lr)

def set_cyclic_lr(optimizer, it, epoch_it, cycles, min_lr, max_lr):
    cycle_length = epoch_it // cycles
    curr_cycle = min(it // cycle_length, cycles-1)
    curr_it = it - cycle_length * curr_cycle

    new_lr = min_lr + 0.5*(max_lr - min_lr)*(1 + np.cos((float(curr_it) / float(cycle_length)) * np.pi))
    set_lr(optimizer, new_lr)

def resample(audio, orig_sr, new_sr, mode="numpy"):
    if orig_sr == new_sr:
        return audio

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    out = librosa.resample(audio, orig_sr, new_sr, res_type='kaiser_fast')

    if mode == "pytorch":
        out = torch.tensor(out)
    return out

class DataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__(module, device_ids, output_device, dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def save_model(model, optimizer, state, path):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # save state dict of wrapped module
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'state': state,  # state of training loop (was 'step')
    }, path)

def load_model(model, optimizer, path, cuda):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        # work-around for loading checkpoints where DataParallel was saved instead of inner module
        from collections import OrderedDict
        model_state_dict_fixed = OrderedDict()
        prefix = 'module.'
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith(prefix):
                k = k[len(prefix):]
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'state' in checkpoint:
        state = checkpoint['state']
    else:
        # older checkpoitns only store step, rest of state won't be there
        state = {'step': checkpoint['step']}
    return state

def load_latest_model_from(model, optimizer, location, cuda):
    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)
    print("load model " + newest_file)
    return load_model(model, optimizer, newest_file, cuda)

def seed_torch(seed=0):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

def alignment(song_pred, lyrics, idx):
    audio_length, num_class = song_pred.shape
    lyrics_int = text2seq(lyrics)
    lyrics_length = len(lyrics_int)

    s = np.zeros((audio_length, 2*lyrics_length+1)) - np.Inf
    opt = np.zeros((audio_length, 2*lyrics_length+1))

    blank = 28

    # init
    s[0][0] = song_pred[0][blank]
    # insert eps
    for i in np.arange(1, audio_length):
        s[i][0] = s[i-1][0] + song_pred[i][blank]

    for j in np.arange(lyrics_length):
        if j == 0:
            s[j+1][2*j+1] = s[j][2*j] + song_pred[j+1][lyrics_int[j]]
            opt[j+1][2*j+1] = 1  # 45 degree
        else:
            s[j+1][2*j+1] = s[j][2*j-1] + song_pred[j+1][lyrics_int[j]]
            opt[j+1][2*j+1] = 2 # 28 degree

        s[j+2][2*j+2] = s[j+1][2*j+1] + song_pred[j+2][blank]
        opt[j+2][2*j+2] = 1  # 45 degree


    for audio_pos in np.arange(2, audio_length):

        for ch_pos in np.arange(1, 2*lyrics_length+1):

            if ch_pos % 2 == 1 and (ch_pos+1)/2 >= audio_pos:
                break
            if ch_pos % 2 == 0 and ch_pos/2 + 1 >= audio_pos:
                break

            if ch_pos % 2 == 1: # ch
                ch_idx = int((ch_pos-1)/2)
                # cur ch -> ch
                a = s[audio_pos-1][ch_pos] + song_pred[audio_pos][lyrics_int[ch_idx]]
                # last ch -> ch
                b = s[audio_pos-1][ch_pos-2] + song_pred[audio_pos][lyrics_int[ch_idx]]
                # eps -> ch
                c = s[audio_pos-1][ch_pos-1] + song_pred[audio_pos][lyrics_int[ch_idx]]
                if a > b and a > c:
                    s[audio_pos][ch_pos] = a
                    opt[audio_pos][ch_pos] = 0
                elif b >= a and b >= c:
                    s[audio_pos][ch_pos] = b
                    opt[audio_pos][ch_pos] = 2
                else:
                    s[audio_pos][ch_pos] = c
                    opt[audio_pos][ch_pos] = 1

            if ch_pos % 2 == 0: # eps
                # cur ch -> ch
                a = s[audio_pos-1][ch_pos] + song_pred[audio_pos][blank]
                # eps -> ch
                c = s[audio_pos-1][ch_pos-1] + song_pred[audio_pos][blank]
                if a > c:
                    s[audio_pos][ch_pos] = a
                    opt[audio_pos][ch_pos] = 0
                else:
                    s[audio_pos][ch_pos] = c
                    opt[audio_pos][ch_pos] = 1

    score = s[audio_length-1][2*lyrics_length]

    # retrive optimal path
    path = []
    x = audio_length-1
    y = 2*lyrics_length
    path.append([x, y])
    while x > 0 or y > 0:
        if opt[x][y] == 1:
            x -= 1
            y -= 1
        elif opt[x][y] == 2:
            x -= 1
            y -= 2
        else:
            x -= 1
        path.append([x, y])

    path = list(reversed(path))
    word_align = []
    path_i = 0

    word_i = 0
    while word_i < len(idx):
        # e.g. "happy day"
        # find the first time "h" appears
        if path[path_i][1] == 2*idx[word_i][0]+1:
            st = path[path_i][0]
            # find the first time " " appears after "h"
            while  path_i < len(path)-1 and (path[path_i][1] != 2*idx[word_i][1]+1):
                path_i += 1
            ed = path[path_i][0]
            # append
            word_align.append([st, ed])
            # move to next word
            word_i += 1
        else:
            # move to next audio frame
            path_i += 1

    return word_align, score



def text2seq(text):
    seq = []
    for c in text.lower():
        idx = string.ascii_lowercase.find(c)
        if idx == -1:
            if c == "'":
                idx = 26
            elif c == " ":
                idx = 27
            else:
                continue # remove unknown characters
        seq.append(idx)
    if len(seq) == 0:
        seq.append(27)
    return np.array(seq)
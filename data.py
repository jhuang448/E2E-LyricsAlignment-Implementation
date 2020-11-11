import h5py

import os
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset, IterableDataset
import glob

from tqdm import tqdm
import DALI as dali_code
from utils import load, write_wav

import logging
# logging.basicConfig(level=logging.DEBUG)

def getDALI(database_path, level, lang, genre):
    dali_annot_path = os.path.join(database_path, 'annot_tismir')
    dali_audio_path = os.path.join(database_path, 'audio')
    dali_data = dali_code.get_the_DALI_dataset(dali_annot_path, skip=[], keep=[])

    # get audio list
    audio_list = os.listdir(os.path.join(dali_audio_path))

    subset = list()
    duration = list()
    total_line_num = 0
    discard_line_num = 0

    for file in audio_list:
        if file.endswith('.mp3') and os.path.exists(os.path.join(dali_annot_path, file[:-4] + '.gz')):
            # get annotation for the current song
            try:
                entry = dali_data[file[:-4]]
                entry_info = entry.info

                # language filter
                if lang is not None and entry_info['metadata']['language'] != lang:
                    continue
                # genre filter
                if genre is not None and genre not in entry_info['metadata']['genres']:
                    continue

                song = {"id": file[:-4], "annot": [], "path": os.path.join(dali_audio_path, file)}
                samples = entry.annotations['annot'][level]
                subset.append(song)

                for sample in samples:
                    sample["duration"] = sample["time"][1] - sample["time"][0]

                    if sample["duration"] > 10.22:
                        print(sample)
                        discard_line_num += 1

                    song["annot"].append(sample)
                    duration.append(sample["duration"])

                    total_line_num += 1

                logging.debug("Successfully loaded {} songs".format(len(subset)))
            except:
                logging.warning("Error loading annotation for song {}".format(file))
                pass

    logging.debug("Scanning {} songs.".format(len(subset)))
    logging.debug("Total line num: {} Discarded line num: {}".format(total_line_num,  discard_line_num))

    return np.array(subset, dtype=object)

def get_dali_folds(database_path, level, lang="english", genre=None):
    dataset = getDALI(database_path, level, lang, genre)

    total_len = len(dataset)
    train_len = np.int(0.8 * total_len)

    train_list = np.random.choice(dataset, train_len, replace=False)
    val_list = [elem for elem in dataset if elem not in train_list]
    logging.debug("First training song: " + str(train_list[0]["id"]) + " " + str(len(train_list[0]["annot"])) + " lines")
    logging.debug("train_list {} songs val_list {} songs".format(len(train_list), len(val_list)))
    return {"train" : train_list, "val" : val_list}

def crop(mix, targets, shapes):
    '''
    Crops target audio to the output shape required by the model given in "shapes"
    '''
    for key in targets.keys():
        if key != "mix":
            targets[key] = targets[key][:, shapes["output_start_frame"]:shapes["output_end_frame"]]
    return mix, targets

def random_amplify(mix, targets, shapes, min, max):
    '''
    Data augmentation by randomly amplifying sources before adding them to form a new mixture
    :param mix: Original mixture
    :param targets: Source targets
    :param shapes: Shape dict from model
    :param min: Minimum possible amplification
    :param max: Maximum possible amplification
    :return: New data point as tuple (mix, targets)
    '''
    residual = mix  # start with original mix
    for key in targets.keys():
        if key != "mix":
            residual -= targets[key]  # subtract all instruments (output is zero if all instruments add to mix)
    mix = residual * np.random.uniform(min, max)  # also apply gain data augmentation to residual
    for key in targets.keys():
        if key != "mix":
            targets[key] = targets[key] * np.random.uniform(min, max)
            mix += targets[key]  # add instrument with gain data augmentation to mix
    mix = np.clip(mix, -1.0, 1.0)
    return crop(mix, targets, shapes)

class LyricsAlignDataset(IterableDataset):
    def __init__(self, dataset, partition, sr, shapes, hdf_dir, in_memory=False, dummy=False):
        '''

        :param dataset:     a list of song with line level annotation
        :param sr:          sampling rate
        :param shapes:      dict, keys: "output_frames", "output_start_frame", "input_frames"
        :param hdf_dir:     hdf5 file
        :param in_memory:   load in memory or not
        :param dummy:       use a subset
        '''

        super(LyricsAlignDataset, self).__init__()

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        if dummy == False:
            self.hdf_file = os.path.join(hdf_dir, partition + ".hdf5")
        else:
            self.hdf_file = os.path.join(hdf_dir, partition + "_dummy.hdf5")

        self.sr = sr
        self.shapes = shapes
        self.hop = (shapes["output_frames"] // 2)
        self.in_memory = in_memory

        # PREPARE HDF FILE

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_file):
            # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # Create HDF file
            with h5py.File(self.hdf_file, "w") as f:
                f.attrs["sr"] = sr

                print("Adding audio files to dataset (preprocessing)...")
                for idx, example in enumerate(tqdm(dataset[partition])):
                    # Load song
                    y, _ = load(example["path"], sr=self.sr, mono=True)

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=y.shape, dtype=y.dtype, data=y)

                    grp.attrs["input_length"] = y.shape[1]

                    annot_num = len(example["annot"])
                    lyrics = [sample["text"].encode() for sample in example["annot"]]
                    times = np.array([sample["time"] for sample in example["annot"]])

                    grp.attrs["annot_num"] = annot_num

                    grp.create_dataset("lyrics", shape=(annot_num, 1), dtype='S100', data=lyrics)
                    grp.create_dataset("times", shape=(annot_num, 2), dtype=times.dtype, data=times)

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_file, "r") as f:
            if f.attrs["sr"] != sr:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate is not as expected. Did you load an out-dated HDF file?")

        # HDF FILE READY

        # SET SAMPLING POSITIONS

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_file, "r") as f:
            # length of song
            lengths = [f[str(song_idx)].attrs["input_length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [( (l - self.shapes["output_frames"]) // self.hop) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]

    def __iter__(self):
        return self

    def __next__(self):

        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_file, 'r', driver=driver)

        while True:

            index = np.random.randint(self.length)

            # Find out which slice of targets we want to read
            song_idx = self.start_pos.bisect_right(index)
            if song_idx > 0:
                index = index - self.start_pos[song_idx - 1]

            # Check length of audio signal
            audio_length = self.hdf_dataset[str(song_idx)].attrs["input_length"]
            annot_num = self.hdf_dataset[str(song_idx)].attrs["annot_num"]
            target_length = self.shapes["output_frames"]

            # Determine position where to start targets
            start_target_pos = index * self.hop
            end_target_pos = start_target_pos + self.shapes["output_frames"]

            # READ INPUTS
            # Check front padding
            start_pos = start_target_pos - self.shapes["output_start_frame"]
            if start_pos < 0:
                # Pad manually since audio signal was too short
                pad_front = abs(start_pos)
                start_pos = 0
            else:
                pad_front = 0

            # Check back padding
            end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
            if end_pos > audio_length:
                # Pad manually since audio signal was too short
                pad_back = end_pos - audio_length
                end_pos = audio_length
            else:
                pad_back = 0

            # read audio and zero padding
            audio = self.hdf_dataset[str(song_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
            if pad_front > 0 or pad_back > 0:
                audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

            # find the lyrics within (start_target_pos, end_target_pos)
            words_start_end_pos = self.hdf_dataset[str(song_idx)]["times"][:]
            first_word_to_include = next(x for x, val in enumerate(list(words_start_end_pos[:, 0]))
                                         if val > start_target_pos/self.sr)
            last_word_to_include = annot_num - next(x for x, val in enumerate(reversed(list(words_start_end_pos[:, 1])))
                                         if val < end_target_pos/self.sr)

            targets = " "
            if first_word_to_include - 1 == last_word_to_include + 1: # the word covers the whole window
                # invalid sample, skip
                targets = None
                continue
            if first_word_to_include <= last_word_to_include: # the window covers word[first:last+1]
                lyrics = self.hdf_dataset[str(song_idx)]["lyrics"][first_word_to_include:last_word_to_include+1]
                lyrics_list = [s[0].decode() for s in list(lyrics)]
                targets = " ".join(lyrics_list)
                targets = " ".join(targets.split())

            return audio, targets

    def __len__(self):
        return self.length

class SeparationDataset(Dataset):
    def __init__(self, dataset, partition, instruments, sr, channels, shapes, random_hops, hdf_dir, audio_transform=None, in_memory=False):
        '''

        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. If True, randomly sample a position from the audio
        '''

        super(SeparationDataset, self).__init__()

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")

        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.in_memory = in_memory
        self.instruments = instruments

        # PREPARE HDF FILE

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_dir):
            # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # Create HDF file
            with h5py.File(self.hdf_dir, "w") as f:
                f.attrs["sr"] = sr
                f.attrs["channels"] = channels
                f.attrs["instruments"] = instruments

                print("Adding audio files to dataset (preprocessing)...")
                for idx, example in enumerate(tqdm(dataset[partition])):
                    # Load mix
                    mix_audio, _ = load(example["mix"], sr=self.sr, mono=(self.channels == 1))

                    source_audios = []
                    for source in instruments:
                        # In this case, read in audio and convert to target sampling rate
                        source_audio, _ = load(example[source], sr=self.sr, mono=(self.channels == 1))
                        source_audios.append(source_audio)
                    source_audios = np.concatenate(source_audios, axis=0)
                    assert(source_audios.shape[1] == mix_audio.shape[1])

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=mix_audio.shape, dtype=mix_audio.dtype, data=mix_audio)
                    grp.create_dataset("targets", shape=source_audios.shape, dtype=source_audios.dtype, data=source_audios)
                    grp.attrs["length"] = mix_audio.shape[1]
                    grp.attrs["target_length"] = source_audios.shape[1]

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_dir, "r") as f:
            if f.attrs["sr"] != sr or \
                    f.attrs["channels"] != channels or \
                    list(f.attrs["instruments"]) != instruments:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate and channel or instruments are not as expected. Did you load an out-dated HDF file?")

        # HDF FILE READY

        # SET SAMPLING POSITIONS

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_dir, "r") as f:
            lengths = [f[str(song_idx)].attrs["target_length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [(l // self.shapes["output_frames"]) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]

    def __getitem__(self, index):
        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r', driver=driver)

        # Find out which slice of targets we want to read
        audio_idx = self.start_pos.bisect_right(index)
        if audio_idx > 0:
            index = index - self.start_pos[audio_idx - 1]

        # Check length of audio signal
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]
        target_length = self.hdf_dataset[str(audio_idx)].attrs["target_length"]

        # Determine position where to start targets
        if self.random_hops:
            start_target_pos = np.random.randint(0, max(target_length - self.shapes["output_frames"] + 1, 1))
        else:
            # Map item index to sample position within song
            start_target_pos = index * self.shapes["output_frames"]

        # READ INPUTS
        # Check front padding
        start_pos = start_target_pos - self.shapes["output_start_frame"]
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # Read and return
        audio = self.hdf_dataset[str(audio_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        targets = self.hdf_dataset[str(audio_idx)]["targets"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            targets = np.pad(targets, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        targets = {inst : targets[idx*self.channels:(idx+1)*self.channels] for idx, inst in enumerate(self.instruments)}

        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio, targets = self.audio_transform(audio, targets)

        return audio, targets

    def __len__(self):
        return self.length
import h5py

import os
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
import glob
import string

from tqdm import tqdm
import DALI as dali_code
from utils import load_example, load, write_wav, load_lyrics, mix_vocal_accompaniment

import logging
# logging.basicConfig(level=logging.DEBUG)

def getDALI(database_path, level, lang, genre, sepa_audio_path=None):
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

                # add separated paths
                if sepa_audio_path is not None:
                    song["vocals"] = os.path.join(sepa_audio_path, file[:-4] + "_vocals.mp3")
                    song["accompaniment"] = os.path.join(sepa_audio_path, file[:-4] + "_accompaniment.mp3")

                    if os.path.exists(song["vocals"]) == False or os.path.exists(song["accompaniment"]) == False:
                        # problematic files (failed at source separation for some reason)
                        print("Separated files not found.", song)
                        song["vocals"] = None
                        song["accompaniment"] = None

                samples = entry.annotations['annot'][level]
                subset.append(song)

                for sample in samples:
                    sample["duration"] = sample["time"][1] - sample["time"][0]

                    if sample["duration"] > 10.22:
                        # print(sample)
                        discard_line_num += 1

                    song["annot"].append(sample)
                    duration.append(sample["duration"])

                    total_line_num += 1

                # logging.debug("Successfully loaded {} songs".format(len(subset)))
            except:
                logging.warning("Error loading annotation for song {}".format(file))
                pass

    logging.debug("Scanning {} songs.".format(len(subset)))
    logging.debug("Total line num: {} Discarded line num: {}".format(total_line_num,  discard_line_num))

    return np.array(subset, dtype=object)

def get_dali_folds(database_path, level, lang="english", genre=None, sepa_audio_path=None):
    dataset = getDALI(database_path, level, lang, genre, sepa_audio_path)

    total_len = len(dataset)
    train_len = np.int(0.8 * total_len)

    train_list = np.random.choice(dataset, train_len, replace=False)
    # test random seed
    print(train_list[0]["id"], train_list[1]["id"], train_list[2]["id"])
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

class LyricsAlignDataset(Dataset):
    def __init__(self, dataset, partition, sr, shapes, hdf_dir,
                 in_memory=False, sepa=False, dummy=False, mute_prob=0.8, aug=False):
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
        hdf_dir = os.path.join(hdf_dir, "sepa={}".format(sepa))
        os.makedirs(hdf_dir, exist_ok=True)
        if dummy == False:
            self.hdf_file = os.path.join(hdf_dir, partition + ".hdf5")
        else:
            self.hdf_file = os.path.join(hdf_dir, partition + "_dummy.hdf5")

        self.sr = sr
        self.shapes = shapes
        self.hop = (shapes["output_frames"] // 2)
        self.in_memory = in_memory
        self.aug = aug
        self.sepa = sepa
        self.mute_prob = mute_prob

        if aug:
            assert(sepa == True)

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
                    if sepa:
                        y, _ = load_example(example, sr=self.sr, mono=True)
                    else:
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
        with h5py.File(self.hdf_file, "r", libver='latest', swmr=True) as f:
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
        self.length_base = self.start_pos[-1]

        if self.aug:
            self.length = self.length_base * 2 # add augmented data
        else:
            self.length = self.length_base

        self.shuffled_buffer = np.arange(self.length)
        self.shuffle_data_list()

    def shuffle_data_list(self):
        np.random.shuffle(self.shuffled_buffer)

    def __getitem__(self, index):

        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_file, 'r', driver=driver)

        while True:
            # Loop until it finds a valid sample

            sepa_flag = (self.aug == False and self.sepa) or (index >= self.length_base)

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

            # find the lyrics within (start_target_pos, end_target_pos)
            words_start_end_pos = self.hdf_dataset[str(song_idx)]["times"][:]
            try:
                first_word_to_include = next(x for x, val in enumerate(list(words_start_end_pos[:, 0]))
                                             if val > start_target_pos/self.sr)
            except StopIteration:
                first_word_to_include = np.Inf

            try:
                last_word_to_include = annot_num - 1 - next(x for x, val in enumerate(reversed(list(words_start_end_pos[:, 1])))
                                             if val < end_target_pos/self.sr)
            except StopIteration:
                last_word_to_include = -np.Inf

            targets = ""
            if first_word_to_include - 1 == last_word_to_include + 1: # the word covers the whole window
                # invalid sample, skip
                targets = None
                index = np.random.randint(self.length)
                continue

            if first_word_to_include <= last_word_to_include: # the window covers word[first:last+1]
                lyrics = self.hdf_dataset[str(song_idx)]["lyrics"][first_word_to_include:last_word_to_include + 1]
                lyrics_list = [s[0].decode() for s in list(lyrics)]
                times_list = self.hdf_dataset[str(song_idx)]["times"][first_word_to_include:last_word_to_include + 1,
                             :] * self.sr - start_pos

                if sepa_flag and audio.shape[0] > 1:
                    audio, lyrics_list = mix_vocal_accompaniment(audio, lyrics_list, times_list, self.mute_prob)
                    # write_wav("{}_{}_after.wav".format(str(song_idx), str(index)), audio, self.sr)

                targets = " ".join(lyrics_list)
                targets = " ".join(targets.split())

            if audio.shape[0] > 1:
                audio = np.sum(audio, axis=0, keepdims=True)
            if pad_front > 0 or pad_back > 0:
                audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

            if len(targets) > 120:
                index = np.random.randint(self.length)
                continue

            seq = self.text2seq(targets)
            break

        return audio, targets, seq

    def text2seq(self, text):
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
        # if len(seq) == 0:
        #     seq.append(28) # insert epsilon for instrumental segments
        return np.array(seq)


    def __len__(self):
        return self.length

class JamendoLyricsDataset(Dataset):
    def __init__(self, sr, shapes, hdf_dir, dataset, jamendo_dir, in_memory=False):
        super(JamendoLyricsDataset, self).__init__()
        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_file = os.path.join(hdf_dir, dataset + ".hdf5")

        self.sr = sr
        self.shapes = shapes
        self.hop = shapes["output_frames"]
        self.in_memory = in_memory

        audio_dir = os.path.join(jamendo_dir, 'mp3')
        lyrics_dir = os.path.join(jamendo_dir, 'lyrics')
        self.audio_list = [file for file in os.listdir(audio_dir) if file.endswith('.mp3')]

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
                for idx, audio_name in enumerate(tqdm(self.audio_list)):

                    # Load song
                    y, _ = load(os.path.join(audio_dir, audio_name), sr=self.sr, mono=True)

                    lyrics, words, idx_in_full = load_lyrics(os.path.join(lyrics_dir, audio_name[:-4]))
                    annot_num = len(words)

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=y.shape, dtype=y.dtype, data=y)

                    grp.attrs["input_length"] = y.shape[1]
                    grp.attrs["audio_name"] = audio_name[:-4]
                    print(len(lyrics))

                    grp.create_dataset("lyrics", shape=(1, 1), dtype='S3000', data=np.array([lyrics.encode()]))
                    grp.create_dataset("idx", shape=(annot_num, 2), dtype=np.int, data=idx_in_full)

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_file, "r", libver='latest', swmr=True) as f:
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
            lengths = [np.int(np.ceil(l / self.hop)) for l in lengths]

        self.lengths = lengths
        self.length = len(lengths)

    def __getitem__(self, index):

        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_file, 'r', driver=driver)

        # select song: index
        # Check length of audio signal
        audio_length = self.hdf_dataset[str(index)].attrs["input_length"]

        # number of chunks for that song
        num_chunk = self.lengths[index]

        chunks = []

        for i in np.arange(num_chunk):
            # Determine position where to start targets
            start_target_pos = i * self.hop
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
                audio = self.hdf_dataset[str(index)]["inputs"][:, start_pos:end_pos].astype(np.float32)
                audio_name = self.hdf_dataset[str(index)].attrs["audio_name"]
                lyrics = self.hdf_dataset[str(index)]["lyrics"][0, 0].decode()
                align_idx = self.hdf_dataset[str(index)]["idx"]
                if pad_front > 0 or pad_back > 0:
                    audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

            chunks.append(audio)

        return chunks, align_idx, (lyrics, audio_name, audio_length)

    def __len__(self):
        return self.length

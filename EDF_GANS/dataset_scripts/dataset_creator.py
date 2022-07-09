import mne
import math
import numpy as np
import librosa
import scipy
from scipy.io import wavfile
import sys
from os.path import isdir, isfile
from progress.bar import ChargingBar

#THE IDEA:
#1. Some stuff like kicking unpaired
#2. We create a big audio (as if a person was listening to the whole file)
#3. Now we have a parallel between edf and our file:
#       a) so we take a labeled sample from edf and some samples after it
#       b) we find the corresponding sample in big audio and to the same thing (maybe we get a different number of samples)
#       c) That is, we got a piece of an audio and a corresponding piece of edf. We can repeat it several times (near our label)
#                                                                                then we move to the next one

#Output: each line consists of concatenated spectrum and flattened EEG data from user's channels

class UserException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message

class Settings():
    def __init__(self) -> None:
        self.audios_dir = None
        self.audios_names = None
        self.big_audio_dir = None
        self.big_audio_name = None
        self.save_big_audio = None
        self.target_channels = None
        self.labeled_ch = None
        self.min_label = None
        self.max_label = None
        self.audio_sr = None
        self.edf_sr = None
        self.audio_framesize = None
        self.edf_framesize = None
        self.hoplength = None
        self.to_the_right = None
        self.with_step = None

    def get_settings(self) -> tuple:
        return (
            self.audios_dir,
            self.audios_names,
            self.big_audio_dir,
            self.big_audio_name,
            self.save_big_audio,
            self.target_channels,
            self.labeled_ch,
            self.min_label,
            self.max_label,
            self.audio_sr,
            self.edf_sr,
            self.audio_framesize,
            self.edf_framesize,
            self.hoplength,
            self.to_the_right,
            self.with_step
        )

#events[1] are labels => check if unpaired. events[0] are data pieces.
def kick_unpaired(events: np.array) -> np.array:
    paired = np.empty((0, 2))
    for i in range(len(events[1]) - 1):
        if events[1][i] == events[1][i+1] + 10:
            paired = np.concatenate((paired, np.array([events[0][i], events[1][i]]).reshape(1, 2)), axis = 0)
            paired = np.concatenate((paired, np.array([events[0][i+1], events[1][i+1]]).reshape(1, 2)), axis = 0)
    return paired

#creates an audio at config.big_audio_dir (as if a person were listening to the whole file)
def create_big_audio(audio_list: list, inner: np.array, length: int, sr_ratio: float) -> np.array:
    big_audio = np.empty(0)
    big_audio = np.append(big_audio, np.zeros(round(sr_ratio * inner[0][0])))
    
    bar = ChargingBar('Creating big audio', max=inner.shape[0] + 1)
    for i in range(inner.shape[0]):
        right_audio = np.array(audio_list[inner[i][1] - config.min_label][0]).astype(float)
        big_audio = np.append(big_audio, right_audio)
        big_audio = np.append(big_audio, np.zeros(round(sr_ratio * (inner[i+1][0] - inner[i][0])) - len(right_audio))) if (i < inner.shape[0] - 1) else big_audio
        bar.next()

    if len(big_audio) < round(length * sr_ratio):
        big_audio = np.append(big_audio, np.zeros(round(length * sr_ratio) - len(big_audio)))
    else:
        big_audio = big_audio[0:round(length * sr_ratio)]
    bar.next()
    bar.finish()
    return big_audio

def edf_to_audio(x: int, sr_ratio: float) -> int:
    return round(sr_ratio * x)

try:
    in_filename = sys.argv[1]
    if in_filename[-4:].lower() != '.edf':
        raise UserException("ERROR: Expected the first argument to be a .edf file")
    out_filename = sys.argv[2] if len(sys.argv) >= 3 else f"{in_filename[:-4]}_dataset.csv"
    if out_filename[-4:].lower() != '.csv':
        raise UserException("ERROR: Expected the second argument to be a .csv file or nothing")

    out_filename_name = out_filename.split("/")[-1]
    out_filename_path = out_filename.replace("/" + out_filename.split("/")[-1], '')
    out_filename_path = "." if out_filename_name == out_filename_path else out_filename_path
    if not isdir(out_filename_path):
        raise UserException("ERROR: output file path doesn't exist")

    just_filename = in_filename.split("/")[-1]

    config_name = sys.argv[3] if len(sys.argv) >= 4 else "config.txt"
    if config_name[-4:] != ".txt":
        raise UserException("ERROR: Expected the third argument to be a .txt file or nothing")
    if not isfile(config_name):
        raise UserException("ERROR: config file path doesn't exist")

    rawedf = mne.io.read_raw_edf(in_filename)
    edf_len = len(rawedf)
except UserException as e:
    print(e)
    exit(-1)
except FileNotFoundError:
    print("ERROR: File not Found")
    exit(-1)
except:
    print("ERROR: Unknown Error")
    exit(-1)


#reading all the settings
try:
    print("Reading config...")
    with open(config_name) as f:
        config = Settings()

        #short audios path (like phonemes)
        config.audios_dir = f.readline().split(" = ")[1].replace('"', '').replace('\n', '')
        config.audios_dir = "." if config.audios_dir == "default" else config.audios_dir
        if not isdir(config.audios_dir):
            raise UserException(f"ERROR: Audios directory ({config.audios_dir}) not found")

        #their names
        config.audios_names = f.readline().split(" = ")[1].replace('"', '').replace('\n', '').split(", ")
        for file in config.audios_names:
            if not isfile(f"{config.audios_dir}/{file}"):
                raise UserException(f"ERROR: Audiofile {file} not found at {config.audios_names}")

        #path for storing a big audio
        config.big_audio_dir = f.readline().split(" = ")[1].replace('"', '').replace('\n', '')
        config.big_audio_dir = "." if config.big_audio_dir == "default" else config.big_audio_dir
        if not isdir(config.audios_dir):
            raise UserException(f"ERROR: Big audio directory ({config.big_audio_dir}) not found")

        #its name
        config.big_audio_name = f.readline().split(" = ")[1].replace('"', '').replace('\n', '')
        config.big_audio_name = (just_filename[:-4] + "_big_audio.wav") if config.big_audio_name == "default" else config.big_audio_name

        #save or discard it?
        config.save_big_audio = bool(f.readline().split(" = ")[1].replace('\n', ''))

        #significant channels
        config.target_channels = f.readline().split(" = ")[1].replace('"', '').replace('\n', '').split(", ")

        #the only special channel with labels (integers)
        config.labeled_ch = f.readline().split(" = ")[1].replace('"', '').replace('\n', '')

        #we're working with all the labels from <config.min_label> to <config.max_label>
        config.min_label = f.readline().split(" = ")[1].replace('\n', '')
        config.min_label = 0 if config.min_label == "default" else int(config.min_label)
        if config.min_label < 0:
            raise UserException("ERROR: min_label < 0")

        config.max_label = f.readline().split(" = ")[1].replace('\n', '')
        config.max_label = 100 if config.max_label == "default" else int(config.max_label)
        if config.max_label < config.min_label:
            raise UserException("ERROR: max_label < min_label")

        #sample rate of a short audio
        config.audio_sr = f.readline().split(" = ")[1].replace('\n', '')
        config.audio_sr = 22050 if config.audio_sr == "default" else int(config.audio_sr)

        config.edf_sr = f.readline().split(" = ")[1].replace('\n', '')
        config.edf_sr = 1006.04 if config.edf_sr == "default" else float(config.edf_sr)

        config.audio_framesize = f.readline().split(" = ")[1].replace('\n', '')
        config.audio_framesize = 222 if config.audio_framesize == "default" else int(config.audio_framesize)

        config.edf_framesize = f.readline().split(" = ")[1].replace('\n', '')
        config.edf_framesize = 10 if config.edf_framesize == "default" else int(config.edf_framesize)

        config.hoplength = f.readline().split(" = ")[1].replace('\n', '')
        config.hoplength = 64 if config.hoplength == "default" else int(config.hoplength)

        #we're working with data from <first_labeled_sample> to <first_labeled_sample + config.to_the_right>
        config.to_the_right = f.readline().split(" = ")[1].replace('\n', '')
        config.to_the_right = 250 if config.to_the_right == "default" else int(config.to_the_right)

        #our data pieces are in
        #[first_labeled_sample, first_labeled_sample + 1*config.with_step],
        #[first_labeled_sample + 1*config.with_step, first_labeled_sample + 2*config.with_step]...
        #and so on
        config.with_step = f.readline().split(" = ")[1].replace('\n', '')
        config.with_step = 25 if config.with_step == "default" else int(config.with_step)

        #print(config.get_settings())
except UserException as e:
    print(e)
    exit(-1)
except:
    print("ERROR: Incorrect options")
    exit(-1)

#MAIN
try:
    data = np.empty((0, edf_len))
    for ch in config.target_channels:
        data = np.concatenate((data, rawedf[ch][0].reshape(1, -1)), axis = 0)
    data = data.T

    print("Detecting events...")
    sys.stdout = open("nul", "w")
    sys.stderr = open("nul", "w")

    events = mne.find_events(rawedf, stim_channel=config.labeled_ch)
    events = np.concatenate((events[:,0].reshape(1, -1), events[:,2].reshape(1, -1)), axis = 0)

    #getting rid of some garbage info from mne
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    paired = kick_unpaired(events).astype(int)

    inner = paired[paired[:,1] >= config.min_label]
    inner = inner[inner[:,1] <= config.max_label]

    print(f"Unfiltered events {np.unique(paired[:, 1])}")
    print(f"Filtered events {np.unique(inner[:, 1])}")

    audio_list = []
    bar = ChargingBar('Loading audios    ', max=len(config.audios_names))
    for sound in config.audios_names:
        audio_list.append(librosa.load(f"{config.audios_dir}/{sound}", sr = config.audio_sr))
        bar.next()
    bar.finish()

    big_audio = create_big_audio(audio_list, inner, edf_len, config.audio_sr / config.edf_sr)
    if config.save_big_audio:
        print(f"Writing \"{config.big_audio_name}\" to \"{config.big_audio_dir}\"...")
        scipy.io.wavfile.write(f"{config.big_audio_dir}/{config.big_audio_name}", config.audio_sr, big_audio)

    x = librosa.stft(big_audio, n_fft=config.audio_framesize, hop_length=config.hoplength, center=False)
    xabsT = np.abs(x).T

    dataset = np.empty((0, config.audio_framesize // 2 + 1 + len(config.target_channels)*config.edf_framesize))
    bar = ChargingBar('Building dataset  ', max=inner.shape[0] * (config.to_the_right // config.with_step))

    #creates a dataset for edf->audio mapping
    for elem in inner:
        left_edf_sample = elem[0]
        for edf_sample in range(left_edf_sample, left_edf_sample + config.to_the_right + 1, config.with_step):
            audio_sample = edf_to_audio(edf_sample, config.audio_sr / config.edf_sr)
            audio_window = math.floor(audio_sample / config.hoplength)

            #Intuition for std config:
            # 22050/1006 = 21.9
            # 222 (audio framesize) / 21.9 = 10 (edf framesize)
            # So, if frequency of audio is k times higher then edf's, then each edf frame corresponds to k audio frames
            # Why 222 and 10? Because 10 is a good number :) And 222 is the result of calculations above
            line = np.concatenate((data[edf_sample : edf_sample + config.edf_framesize].reshape(-1), xabsT[audio_window].reshape(-1)), axis = 0).reshape(1, -1)
            dataset = np.append(dataset, line, axis = 0)

            bar.next()

    bar.finish()
    print(f"Writing \"{out_filename_name}\" to \"{out_filename_path}\"...")
    np.savetxt(out_filename, dataset, fmt = '%.6f', delimiter = ';')
    print("Success!\n")
except ValueError:
    print("ERROR: Value Error (apparently, you've mentioned a channel which doesn't exist)")
    exit(-1)
except:
    print("ERROR: Unknown Error")
    exit(-1)


import os
import librosa
import numpy as np

#将目录下的所有文件放到列表中去
def load_file(dir):
    file_list = list()
    for filename in os.listdir(dir):
        file_list.append(os.path.join(dir, filename))
    return file_list

#   导入训练集数据，每一个训练集文件都是一个双声道的音频文件，
#   其中，第一个声道存的是背景音乐，第二个声道存的是纯人声，
#   我们需要三组数据，第一组是将双声道转成单声道的数据，即让背景音乐和人声混合在一起
#   第二组数据是纯背景音乐，第三组数据是纯人声数据
def load_wavs(filenames, sr):
    wavs_mono = list()
    wavs_music = list()
    wavs_voice = list()
    #读取wav文件，首先要求源文件是有双声道的音频文件，一个声道存的是背景音乐，另一个声道存的是纯人声
    #然后，将音频转成单声道，存入 wavs_mono
    #将背景音乐存入 wavs_music，
    #将纯人声存入 wavs_voice
    for filename in filenames:
        wav, _ = librosa.load(filename, sr = sr, mono = False)
        assert (wav.ndim == 2) and (wav.shape[0] == 2), '要求WAV文件有两个声道!'

        wav_mono = librosa.to_mono(wav) * 2
        wav_music = wav[0, :]
        wav_voice = wav[1, :]
        wavs_mono.append(wav_mono)
        wavs_music.append(wav_music)
        wavs_voice.append(wav_voice)

    return wavs_mono, wavs_music, wavs_voice

#通过短时傅里叶变换将声音转到频域
def wavs_to_specs(wavs_mono, wavs_music, wavs_voice, n_fft = 1024, hop_length = None):

    stfts_mono = list()
    stfts_music = list()
    stfts_voice = list()

    for wav_mono, wav_music, wav_voice in zip(wavs_mono, wavs_music, wavs_voice):
        stft_mono = librosa.stft(wav_mono, n_fft = n_fft, hop_length = hop_length)
        stft_music = librosa.stft(wav_music, n_fft = n_fft, hop_length = hop_length)
        stft_voice = librosa.stft(wav_voice, n_fft = n_fft, hop_length = hop_length)
        stfts_mono.append(stft_mono)
        stfts_music.append(stft_music)
        stfts_voice.append(stft_voice)

    return stfts_mono, stfts_music, stfts_voice

#stfts_mono：单声道stft频域数据
#stfts_music：背景音乐stft频域数据
#stfts_music：人声stft频域数据
#batch_size：batch大小
#sample_frames：获取多少帧数据
def get_next_batch(stfts_mono, stfts_music, stfts_voice, batch_size = 64, sample_frames = 8):

    stft_mono_batch = list()
    stft_music_batch = list()
    stft_voice_batch = list()

    #随即选择batch_size个数据
    collection_size = len(stfts_mono)
    collection_idx = np.random.choice(collection_size, batch_size, replace = True) #True表示可以取相同数字；

    for idx in collection_idx:
        stft_mono = stfts_mono[idx]
        stft_music = stfts_music[idx]
        stft_voice = stfts_voice[idx]
        #有多少帧
        num_frames = stft_mono.shape[1]
        assert  num_frames >= sample_frames
        #随机获取sample_frames帧数据
        start = np.random.randint(num_frames - sample_frames + 1)
        end = start + sample_frames

        stft_mono_batch.append(stft_mono[:,start:end])
        stft_music_batch.append(stft_music[:,start:end])
        stft_voice_batch.append(stft_voice[:,start:end])

    #将数据转成np.array，再对形状做一些变换
    # Shape: [batch_size, n_frequencies, n_frames]
    stft_mono_batch = np.array(stft_mono_batch)
    stft_music_batch = np.array(stft_music_batch)
    stft_voice_batch = np.array(stft_voice_batch)
    # Shape for RNN: [batch_size, n_frames, n_frequencies]
    data_mono_batch = stft_mono_batch.transpose((0, 2, 1))
    data_music_batch = stft_music_batch.transpose((0, 2, 1))
    data_voice_batch = stft_voice_batch.transpose((0, 2, 1))

    return data_mono_batch, data_music_batch, data_voice_batch

#通过短时傅里叶变换后的结果是复数的，而我们训练时，
#只需要考虑频率部分就可以了，所以将频率和相位分离出来
def separate_magnitude_phase(data):
    return np.abs(data), np.angle(data)

#根据振幅和相位，得到复数，
#信号s(t)乘上e^(j*phases)表示信号s(t)移动相位phases
def combine_magnitude_phase(magnitudes, phases):
    return magnitudes * np.exp(1.j * phases)
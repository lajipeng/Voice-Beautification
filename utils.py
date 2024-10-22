import os
import librosa
import numpy as np

#将目录下的所有文件放到列表中去
def load_file(dir):
    file_list = list()
    for filename in os.listdir(dir):
        file_list.append(os.path.join(dir, filename))
    return file_list

def load_label(dir, sr):
    wavs_label = ['label1', 'label2', 'label3']
    for filename in os.listdir(dir):
        if filename.startswith('label1'):
            wav, _ = librosa.load(os.path.join(dir, filename), sr = sr, mono = False)
            wavs_label[0] = wav
        elif filename.startswith('label2'):
            wav, _ = librosa.load(os.path.join(dir, filename), sr = sr, mono = False)
            wavs_label[1] = wav
        else:
            wav, _ = librosa.load(os.path.join(dir, filename), sr = sr, mono = False)
            wavs_label[2] = wav
    return wavs_label            

#   导入训练集数据，每一个训练集文件都是一个双声道的音频文件，
#   其中，第一个声道存的是背景音乐，第二个声道存的是纯人声，
#   我们需要三组数据，第一组是将双声道转成单声道的数据，即让背景音乐和人声混合在一起
#   第二组数据是纯背景音乐，第三组数据是纯人声数据
def load_wavs(filenames,wavs_label, sr):
    mono = list()
    label = list()
    #读取wav文件，首先要求源文件是有双声道的音频文件，一个声道存的是背景音乐，另一个声道存的是纯人声
    #然后，将音频转成单声道，存入 wavs_mono
    #wavs_label为该段数据对应的label
    for filename in filenames:
        if filename.endswith("label1.wav"):
            wav_label = wavs_label[0]
        elif filename.endswith("label2.wav"):
            wav_label = wavs_label[1]
        else:
            wav_label = wavs_label[2]

        wav_mono, _ = librosa.load(filename, sr = sr, mono = False)

        mono.append(wav_mono)
        label.append(wav_label)

    return mono, label

#通过短时傅里叶变换将声音转到频域
def wavs_to_specs(wavs_mono, wavs_label, n_fft = 1024, hop_length = None):

    stfts_mono = list()
    stfts_label = list()

    for wav_mono, wav_label in zip(wavs_mono, wavs_label):
        stft_mono = librosa.stft(wav_mono, n_fft = n_fft, hop_length = hop_length)
        stft_label = librosa.stft(wav_label, n_fft = n_fft, hop_length = hop_length)
        stfts_mono.append(stft_mono)
        stfts_label.append(stft_label)

    return stfts_mono, stfts_label

#stfts_mono：用户stft频域数据
#stfts_label：标签stft频域数据
#batch_size：batch大小
#sample_frames：获取多少帧数据
def get_next_batch(stfts_mono, stfts_label, batch_size = 64, sample_frames = 8):

    # stft_mono_batch = list()
    stft_mono_batch = np.zeros([64,513,10],dtype=np.complex64)
    stft_label_batch = np.zeros([64,513,10],dtype=np.complex64)

    #随即选择batch_size个数据
    collection_size = len(stfts_mono)
    collection_idx = np.random.choice(collection_size, batch_size, replace = True) #True表示可以取相同数字；
    i = 0
    for idx in collection_idx:
        stft_mono = stfts_mono[idx]
        stft_label = stfts_label[idx]
        #有多少帧
        num_frames = min(stft_mono.shape[1],stft_label.shape[1])
        assert  num_frames >= sample_frames
        #随机获取sample_frames帧数据
        start = np.random.randint(num_frames - sample_frames + 1)
        end = start + sample_frames

        # stft_mono_batch.append(stft_mono[:,start:end])
        stft_mono_batch[i,:,:] =  stft_mono[:,start:end]
        stft_label_batch[i,:,:] =  stft_label[:,start:end]
        i += 1
        

    #将数据转成np.array，再对形状做一些变换
    # Shape: [batch_size, n_frequencies, n_frames]
    # stft_mono_batch = np.array(stft_mono_batch)
    # stft_label_batch = np.array(stft_label_batch)
    # Shape for RNN: [batch_size, n_frames, n_frequencies]
    data_mono_batch = stft_mono_batch.transpose((0, 2, 1))
    data_label_batch = stft_label_batch.transpose((0, 2, 1))

    return data_mono_batch, data_label_batch

#通过短时傅里叶变换后的结果是复数的，而我们训练时，
#只需要考虑频率部分就可以了，所以将频率和相位分离出来
def separate_magnitude_phase(data):
    return np.abs(data), np.angle(data)

#根据振幅和相位，得到复数，
#信号s(t)乘上e^(j*phases)表示信号s(t)移动相位phases
def combine_magnitude_phase(magnitudes, phases):
    return magnitudes * np.exp(1.j * phases)
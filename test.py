#encoding:utf-8

import librosa
import os
import numpy as np
from model import SVMRNN
import argparse
import sys
from utils import separate_magnitude_phase, combine_magnitude_phase



def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    dataset_sr = args.dataset_sr
    model_dir = args.model_dir
    dropout_rate = args.dropout_rate

    #如果输入目录不存在，返回错误
    if not os.path.exists(input_dir):
        raise NameError('音频输入文件夹"./songs/input"不存在！')

    #输出文件夹不存在，则创建
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    #找到要分离背景音乐和人声的音频文件
    song_filenames = list()
    for file in os.listdir(input_dir):
        if file.endswith('.mp3'):
            song_filenames.append(os.path.join(input_dir, file))

    #加载输入音频文件
    wavs_mono = list()
    for filename in song_filenames:
        wav_mono, _ = librosa.load(filename, sr=dataset_sr, mono=True)
        wavs_mono.append(wav_mono)

    # 用于短时傅里叶变换，窗口大小
    n_fft = 1024
    # 步幅;帧移对应卷积中的stride;
    hop_length = n_fft // 4
    # 用于创建rnn节点数
    num_hidden_units = [1024, 1024, 1024, 1024, 1024]

    #将其转到频域
    stfts_mono = list()
    for wav_mono in wavs_mono:
        stft_mono = librosa.stft(wav_mono, n_fft = n_fft, hop_length = hop_length)
        stfts_mono.append(stft_mono.transpose())

    #初始化神经网络
    model = SVMRNN(num_features = n_fft // 2 + 1, num_hidden_units = num_hidden_units)
    #导入模型
    model.load(file_dir = model_dir)

    for wav_filename, wav_mono, stft_mono in zip(song_filenames, wavs_mono, stfts_mono):
        wav_filename_base = os.path.basename(wav_filename)
        #单声道音频文件
        wav_mono_filename = wav_filename_base.split('.')[0] + '_mono.wav'
        #分离后的背景音乐音频文件
        wav_music_filename = wav_filename_base.split('.')[0] + '_music.wav'
        #分离后的人声音频文件
        wav_voice_filename = wav_filename_base.split('.')[0] + '_voice.wav'

        #要保存的文件的相对路径
        wav_mono_filepath = os.path.join(output_dir, wav_mono_filename)
        wav_music_hat_filepath = os.path.join(output_dir, wav_music_filename)
        wav_voice_hat_filepath = os.path.join(output_dir, wav_voice_filename)

        print('Processing %s ...' % wav_filename_base)

        stft_mono_magnitude, stft_mono_phase = separate_magnitude_phase(data = stft_mono)
        stft_mono_magnitude = np.array([stft_mono_magnitude])

        y_music_pred, y_voice_pred = model.test(x_mixed_src = stft_mono_magnitude, dropout_rate = dropout_rate)

        # 根据振幅和相位，转为复数，用于下面的逆短时傅里叶变换
        y_music_stft_hat = combine_magnitude_phase(magnitudes = y_music_pred[0], phases = stft_mono_phase)
        y_voice_stft_hat = combine_magnitude_phase(magnitudes = y_voice_pred[0], phases = stft_mono_phase)

        y_music_stft_hat = y_music_stft_hat.transpose()
        y_voice_stft_hat = y_voice_stft_hat.transpose()

        #逆短时傅里叶变换，将数据从频域转到时域
        y_music_hat = librosa.istft(y_music_stft_hat, hop_length = hop_length)
        y_voice_hat = librosa.istft(y_voice_stft_hat, hop_length = hop_length)

        #保存数据
        librosa.output.write_wav(wav_mono_filepath, wav_mono, dataset_sr)
        librosa.output.write_wav(wav_music_hat_filepath, y_music_hat, dataset_sr)
        librosa.output.write_wav(wav_voice_hat_filepath, y_voice_hat, dataset_sr)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, help='待测试的音频文件的文件夹，存放MP3文件', default='./songs/input')
    parser.add_argument('--output_dir', type=str, help='声乐分离后的视频文件目录，为WAV格式', default='./songs/output')
    parser.add_argument('--model_dir', type=str, help='模型保存的文件夹', default='./model')
    parser.add_argument('--model_filename', type=str, help='模型保存的文件名', default='svmrnn.ckpt')
    parser.add_argument('--dataset_sr', type=int, help='数据集音频文件的采样率', default=16000)
    parser.add_argument('--dropout_rate', type=float, help='dropout率', default=0.95)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
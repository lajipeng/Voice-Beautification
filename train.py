#encoding:utf-8
from model import SVMRNN
import os
import sys
import argparse
from utils import load_file, load_wavs, load_label, wavs_to_specs, get_next_batch, separate_magnitude_phase, load_label
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#   训练模型，需要做以下事情
#1. 导入需要训练的数据集文件路径，存到列表中即可
#2. 导入训练集数据，每一个训练集文件都是一个双声道的音频文件，
#3. 通过上一步获取的数据都是时域的，我们要通过短时傅里叶变换将声音数据转到频域
#4. 初始化网络模型
#5. 获取mini-batch数据，开始进行迭代训练

def main(args):

    #先看数据集数据是否存在
    if not os.path.exists(args.dataset_train_dir) or not os.path.exists(args.dataset_validate_dir):
        raise NameError('数据集路径"./dataset/MIR-1K/Wavfile"或"./dataset/MIR-1K/UndividedWavfile"不存在!')

    # 1. 导入需要训练的数据集文件路径，存到列表中即可
    train_file_list = load_file(args.dataset_train_dir)
    valid_file_list = load_file(args.dataset_validate_dir)

    # 数据集的采样率
    mir1k_sr = args.dataset_sr
    # 用于短时傅里叶变换，窗口大小
    n_fft = 1024
    # 步幅;帧移对应卷积中的stride;
    hop_length = n_fft // 4

    # Model parameters
    # 学习率
    learning_rate = args.learning_rate

    # 用于创建rnn节点数
    num_hidden_units = [1024, 1024, 1024, 1024, 1024]
    # batch 长度
    batch_size = args.batch_size
    # 获取多少帧数据
    sample_frames = args.sample_frames
    # 训练迭代次数
    iterations = args.iterations
    # dropout
    dropout_rate = args.dropout_rate

    # 模型保存路径
    model_dir = args.model_dir
    model_filename = args.model_filename

    #导入训练数据集的wav数据,
    #wavs_mono_train存的是单声道，wavs_label_train 存的是标签
    label_train = load_label(args.dataset_label_dir,sr = mir1k_sr)
    label_test = load_label(args.dataset_label_dir,sr = mir1k_sr)
    wavs_mono_train, wavs_label_train = load_wavs(filenames = train_file_list, wavs_label = label_train, sr = mir1k_sr)
    # 通过短时傅里叶变换将声音转到频域
    stfts_mono_train, stfts_label_train = wavs_to_specs(
        wavs_mono = wavs_mono_train, wavs_label = wavs_label_train, n_fft = n_fft, 
        hop_length = hop_length)

    # 跟上面一样，只不过这里是测试集的数据
    wavs_mono_valid, wavs_label_valid = load_wavs(filenames = valid_file_list, wavs_label = label_test, sr=mir1k_sr)
    stfts_mono_valid, stfts_label_valid = wavs_to_specs(
        wavs_mono = wavs_mono_valid, wavs_label = wavs_label_valid, n_fft=n_fft,
        hop_length = hop_length)
    
    #初始化模型
    model = SVMRNN(num_features = n_fft // 2 + 1, num_hidden_units = num_hidden_units)

    # 加载模型，如果没有模型，则初始化所有变量
    startepo = model.load(file_dir = model_dir)

    print('startepo:' + str(startepo))

    #开始训练
    for i in (range(iterations)):
        #从模型中断处开始训练
        if i < startepo:
            continue

        # 获取下一batch数据
        data_mono_batch, data_label_batch = get_next_batch(
            stfts_mono = stfts_mono_train, stfts_label = stfts_label_train,
            batch_size = batch_size, sample_frames = sample_frames)

        #获取频率值
        x_mixed_src, _ = separate_magnitude_phase(data = data_mono_batch)
        y_label_src, _ = separate_magnitude_phase(data = data_label_batch)

        #送入神经网络，开始训练
        train_loss = model.train(x_mixed_src = x_mixed_src, y_label_src = y_label_src,
                                 learning_rate = learning_rate, dropout_rate = dropout_rate)

        if i % 10 == 0:
            print('Step: %d Train Loss: %f' %(i, train_loss))

        if i % 200 == 0:
            #这里是测试模型准确率的
            print('==============================================')
            data_mono_batch, data_label_batch = get_next_batch(
                stfts_mono = stfts_mono_valid, stfts_label = stfts_label_valid,
                batch_size = batch_size, sample_frames = sample_frames)

            x_mixed_src, _ = separate_magnitude_phase(data = data_mono_batch)
            y_label_src, _ = separate_magnitude_phase(data = data_label_batch)

            y_sing_src_pred, validate_loss = model.validate(x_mixed_src = x_mixed_src,
                    y_label_src = y_label_src, dropout_rate = dropout_rate)
            print('Step: %d Validation Loss: %f' %(i, validate_loss))
            print('==============================================')


        if i % 200 == 0:
            model.save(directory = model_dir, filename = model_filename, global_step=i)

#可以通过命令设置的参数：
#dataset_dir : 数据集路径
#model_dir ： 模型保存的文件夹
#model_filename : 模型保存的文件名
#dataset_sr : 数据集音频文件的采样率
#learning_rate ： 学习率
#batch_size : 小批量训练数据的长度
#sample_frames ： 每次训练获取多少帧数据
#iterations ： 训练迭代次数
#dropout_rate ： dropout率
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_train_dir', type=str, help='数据集训练数据路径', default='./data/train_data')
    parser.add_argument('--dataset_validate_dir', type=str, help='数据集验证数据路径', default='./data/test_data')
    parser.add_argument('--dataset_label_dir', type=str, help='数据集标签数据路径', default='./data/label_data')
    parser.add_argument('--model_dir', type=str, help='模型保存的文件夹', default='model')
    parser.add_argument('--model_filename', type=str, help='模型保存的文件名', default='svmrnn.ckpt')
    parser.add_argument('--dataset_sr', type=int, help='数据集音频文件的采样率', default=16000)
    parser.add_argument('--learning_rate', type=float, help='学习率', default=0.0001)
    parser.add_argument('--batch_size', type=int, help='小批量训练数据的长度', default=64)
    parser.add_argument('--sample_frames', type=int, help='每次训练获取多少帧数据', default=10)
    parser.add_argument('--iterations', type=int, help='训练迭代次数', default=30000)
    parser.add_argument('--dropout_rate', type=float, help='dropout率', default=0.95)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
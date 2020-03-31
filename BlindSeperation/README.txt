准备：
在./dataset目录中放入MIR-1K数据集，作为训练样本，
在./songs/input目录中放入待分离的音频文件，作为测试样本。

解释：
train.py训练模型，并保存在./model路径下，
test.py读取./model下保存的模型，进行检验，
test.py将./songs/input目录下的音频分离、保存到./songs/output目录下。

运行test.py时可能会遇到audioread.NoBackendError，参考以下博客解决：
https://blog.csdn.net/u014742995/article/details/84898901

另外，本次训练目标是30000次，但是当训练次数达到10550次时由于内存溢出，导致训练终止；
但是保存下来的前面10400次训练模型仍然是可以使用的。（本模型测试产生的音频分离文件在./songs/output下，效果可自行检阅）
当然，为了提高模型的性能，仍然建议在有条件的机器上重新运行train.py，得到新的模型。

感谢代码的提供者：
https://blog.csdn.net/rookie_wei/article/details/87831133
from scipy import fft
from scipy.io import wavfile
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import numpy as np

'''
(sample_rate, X) = wavfile.read("D:/genres/blues/converted/blues.00000.au.wav")
print(sample_rate, X.shape)

plt.figure(figsize=(10, 4), dpi=80)
plt.xlabel("time")
plt.ylabel("frequnecy")
plt.grid(True, linestyle='-', color='0.75')
specgram(X, Fs=sample_rate, xextent=(0, 30))
plt.show()

def plotSpec(g, n):
    rad = "D:/genres/"+g+"/converted/"+g+'.'+n+'.au.wav'
    sasmple_rate, X = wavfile._array_tofile((rad))
    fft_features = abs(fft(X, sample_rate)[:1000])
    sad = "d:/trainset/"+g+'.'+str(n).zfill(5)+".fft"
    np.save(sad, fft_features)
    '''

# 准备音乐数据，把音乐文件一个个的去使用傅里叶变换，并且把傅里叶变换之后的结果落地保存
# 提取特征


def create_fft(g, n):
    rad = "D:/genres/"+g+"/converted/"+g+'.'+str(n).zfill(5)+'.au.wav'
    sample_rate, X = wavfile.read(rad)
    fft_features = abs(fft(X)[:1000])
    sad = "d:/trainset/"+g+'.'+str(n).zfill(5)+".fft"
    np.save(sad, fft_features)


genre_list = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal']
for g in genre_list:
    for n in range(100):
        create_fft(g, n)

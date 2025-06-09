# coding: utf-8
# 短時間フーリエ変換のコードは下記を参考にした
#  http://yukara-13.hatenablog.com/entry/2013/11/17/210204
from numpy import ceil, complex64, float64, hamming, zeros
from scipy.fftpack import fft, ifft
import numpy as np
import wave

import librosa

def stft(x, win, step):
    """
    短時間フーリエ変換（STFT）を計算する．
    
    xで与えられた音声波形に対して，窓関数winで短時間に切り出し，
    FFTを適用することを，step点だけシフトして繰り返す．


    Parameters
    ----------
    x : numpy.ndarray
        入力信号(モノラル)
    win : numpy.ndarray
        窓関数
    step : int
        シフト幅
    
    Returns
    -------
    numpy.ndarray
       STFTの結果．
       shape が (frames, bins) のアレイ．
       framesはフレーム数，binsはFFTのポイント数．
       要素はcomplex64型の値である．
    """
    X = librosa.stft(x, n_fft=len(win), hop_length=step, win_length=len(win), window="hann", center=True, pad_mode='constant')
    return X.T

    # L = len(x)  # 入力信号の長さ
    # N = len(win)  # 窓幅、つまり切り出す幅
    # M = int(ceil(float(L - N + step) / step))  # スペクトログラムの時間フレーム数

    # # import ipdb; ipdb.set_trace()

    # new_x = zeros(N + ((M - 1) * step), dtype=float64)
    # new_x[:L] = x  # 信号をいい感じの長さにする

    # X = zeros([M, N], dtype=complex64)  # スペクトログラムの初期化(複素数型)
    # for m in range(M):
    #     start = step * m
    #     X[m, :] = fft(new_x[start:start + N] * win)
    # return X


def stft_multi_channel(x, win, step):
    """
    マルチチャネルのSTFT．

    各チャネルに独立に stft を適用する．

    Parameters
    ----------
    x : numpy.ndarray
        shape が (channels, samples)の波形データ．
        channelsはチャネル数．samplesはサンプル数を表す．
    win : numpy.ndarray
        窓関数
    step : シフト幅

    Returns
    -------
    numpy.ndarray
        STFTの結果．
        shape が(channels, frames, bins)のデータを返す．
        channelsはチャネル数，framesはフレーム数，binsはFFTのポイント数．
        要素はcomplex64型の値．

    See Also
    --------
    stft : モノラルの短時間フーリエ変換
    """
    num_channel, _ = x.shape
    Xs = []
    for i in range(num_channel):
        Xs.append(stft(x[i], win, step))
    X = np.stack(Xs, axis=0)
    return X


def istft(X, win, step):
    """
    短時間逆フーリエ変換（ISTFT）．


    Parameters
    ----------
    X : numpy.ndarray
        複素スペクトログラムが並んだデータ．
        stft の出力と同等な形状をしている必要がある．
    win : numpy.ndarray
        窓関数
    step : int
        シフト幅

    Returns
    -------
    numpy.ndarray
        モノラルの音声データ

    See Also
    --------
    stft : 短時間フーリエ変換
    """
    X = X.T
    return librosa.istft(X, hop_length=step, win_length=len(win), window="hann", center=True)

    # M, N = X.shape
    # assert (len(win) == N), "FFT length and window length are different."

    # L = (M - 1) * step + N
    # # x = zeros(L, dtype=float64)
    # x = np.random.normal(size=(L,)) * 10
    # wsum = zeros(L, dtype=float64)
    # for m in range(M):
    #     start = step * m

    #     # 滑らかな接続
    #     x[start:start + N] = x[start:start + N] + ifft(X[m, :]).real * win
    #     wsum[start:start + N] += win**2
    # # 窓分のスケール合わせ
    # pos = (wsum != 0)
    # x[pos] /= wsum[pos]

    # return x


def istft_multi_channel(X, win, step):
    """
    マルチチャネルのISTFT．

    Parameters
    ----------
    X : numpy.ndarray
        マルチチャネルの複素スペクトログラムが並んだデータ．
        stft_multi_channel の出力と同等な形状をしている必要がある．

    win : numpy.ndarray
        窓関数
    step : int
        シフト幅

    Returns
    -------
    numpy.ndarray
        マルチチャネルのの音声データ．
    
    See Also
    --------
    istft : 短時間逆フーリエ変換
    stft_multi_channel : マルチチャネルの短時間フーリエ変換
    """
    num_channel, _, _ = X.shape
    xs = []
    for i in range(num_channel):
        xs.append(istft(X[i], win, step))
    x = np.stack(xs, axis=0)
    return x


def safia_hard(X):
    """
    SAFIAをハードに適用する．

    与えられたマルチチャネルのスペクトログラムの各周波数ビンに対して，
    最も大きなチャネルの値のみを残し，他は全て0にする．

    
    Parameters
    ----------
    X : numpy.ndarray
        マルチチャネルのスペクトログラム

    Returns
    -------
    numpy.ndarray
        入力と同じ形式のマルチチャネルのスペクトログラム．
        ただし，不要な周波数ビンのスペクトルの値は0になっている．

    See Also
    --------
    stft_multi_channel : マルチチャネルの短時間フーリエ変換
    """
    num_channels, _, _ = X.shape
    max_indices = abs(X).argmax(axis=0)
    Xs = []
    for i in range(num_channels):
        Xout = np.zeros(X.shape[1:], dtype=X.dtype)
        flag = max_indices == i
        Xout[flag] = X[i, flag]
        Xs.append(Xout)
    return np.stack(Xs, axis=0)


def apply_safia(x):
    """
    マルチチャネルの音声にSAFIAを適用する．
        
    Parameters
    ----------
    x : numpy.ndarray
        int16型で，shape が (channels, samples) の波形データ．
        channelsはチャネル数，samplesはサンプル数．

    Returns
    -------
    numpy.ndarray
        x と同じ形状でSAFIAが適用された波形データ．
    """
    fft_len = 512
    win = hamming(fft_len)
    step = fft_len // 4

    x = x.astype(float)
    x /= 32768.0  # 正規化

    X = stft_multi_channel(x, win, step)
    X_safia = safia_hard(X)
    x_safia = istft_multi_channel(X_safia, win, step)
    # 通常，元のデータよりも長くなっている（窓関数分？）
    x_safia = x_safia[:, :x.shape[1]]

    x_safia *= 32768.0  # 復元
    x_safia = np.clip(x_safia, -32768, 32767)

    return np.int16(x_safia)


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 3:
        print("safia.py outdir infiles")
        sys.exit(-1)

    outdir = sys.argv[1]
    infiles = sys.argv[2:]

    wavdata_list = []
    for infile in infiles:
        wf = wave.open(infile, 'r')
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        data = wf.readframes(nframes)
        wf.close()
        # サンプル幅に応じて型を決定
        if sampwidth == 2:
            dtype = 'int16'
        elif sampwidth == 1:
            dtype = 'int8'
        elif sampwidth == 4:
            dtype = 'int32'
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        x = np.frombuffer(data, dtype)
        # モノラル→(1, サンプル数)に
        if channels == 1:
            x = x[np.newaxis, :]
        else:
            x = x.reshape(channels, -1)
        # サンプリングレートが16kHzでなければリサンプリング
        if framerate != 16000:
            x_resampled = []
            for ch in range(x.shape[0]):
                x_resampled.append(librosa.resample(x[ch].astype(float), orig_sr=framerate, target_sr=16000))
            # 長さを揃える
            minlen = min([len(xx) for xx in x_resampled])
            x = np.stack([xx[:minlen] for xx in x_resampled], axis=0)
        # int16でなければ変換
        if x.dtype != np.int16:
            # 16bit幅にスケーリング
            if x.dtype == np.int8:
                x = (x.astype(np.float32) / 128.0 * 32767.0).astype(np.int16)
            elif x.dtype == np.int32:
                x = (x.astype(np.float32) / 2147483648.0 * 32767.0).astype(np.int16)
            else:
                x = x.astype(np.int16)
        wavdata_list.append(x)

    # チャネル数を揃える（最小チャネル数に合わせる）
    min_channels = min([xx.shape[0] for xx in wavdata_list])
    wavdata_list = [xx[:min_channels] for xx in wavdata_list]
    # サンプル数を揃える（最小長に合わせる）
    min_len = min([xx.shape[1] for xx in wavdata_list])
    wavdata_list = [xx[:, :min_len] for xx in wavdata_list]
    x = np.stack(wavdata_list, axis=0)
    x_safia = apply_safia(x)

    for i, infile in enumerate(infiles):
        outfile = os.path.join(outdir, os.path.basename(infile))
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        
        wf = wave.open(outfile, 'w')
        wf.setnchannels(channels)
        wf.setframerate(16000)
        wf.setsampwidth(2)
        wf.writeframes(x_safia[i].tobytes())
        wf.close()

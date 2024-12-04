import numpy as np

def my_rms(emg_window):
    return np.sqrt(np.mean(emg_window ** 2))


def get_rms(emg, window_len, step_len, fs):
    """
    使用滑动窗口方法从EMG数据中提取RMS特征。

    参数:
    - emg: 一个形状为 (Nsample, Nchannel) 的二维numpy数组，每列是一个EMG数据通道。
    - window_len: 滑动窗口的长度（以秒为单位）。
    - step_len: 滑动窗口的步长（以秒为单位）。
    - fs: 采样频率（Hz）。

    返回:
    - rms: 提取的RMS特征，形状为 (num_windows, Nchannel)。
    """
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    Nsample, Nchannel = emg.shape

    # 计算窗口的数量
    num_windows = (Nsample - window_sample) // step_sample + 1
    rms_features = np.zeros((num_windows, Nchannel))

    # 在信号上滑动窗口并计算每个通道的RMS
    for i in range(0, Nsample - window_sample + 1, step_sample):
        window = emg[i:i + window_sample, :]  # 获取窗口（形状为 window_sample x Nchannel）
        rms_features[i // step_sample, :] = np.apply_along_axis(my_rms, 0, window)  # 计算每个通道的RMS

    return rms_features


# 假设 my_wl 在其他地方定义并计算给定窗口的波形长度
def my_wl(emg_window, fs):
    # 波形长度计算的占位符
    # 实际实现将取决于WL的具体算法。
    return np.sum(np.abs(np.diff(emg_window)))  # 示例计算（波形长度）

def get_wl(emg, window_len, step_len, fs):
    """
    使用滑动窗口方法从EMG数据中提取波形长度特征。
    
    参数:
    - emg: 一个形状为 (Nsample, Nchannel) 的二维numpy数组，每列是一个EMG数据通道。
    - window_len: 滑动窗口的长度（以秒为单位）。
    - step_len: 滑动窗口的步长（以秒为单位）。
    - fs: 采样频率（Hz）。
    
    返回:
    - wl: 提取的波形长度特征，形状为 (num_windows, Nchannel)。
    """
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    Nsample, Nchannel = emg.shape
    
    # 计算窗口的数量
    num_windows = (Nsample - window_sample) // step_sample + 1
    wl_features = np.zeros((num_windows, Nchannel))
    
    # 在信号上滑动窗口并计算每个通道的波形长度
    for i in range(0, Nsample - window_sample + 1, step_sample):
        window = emg[i:i + window_sample, :]  # 获取窗口（形状为 window_sample x Nchannel）
        wl_features[i // step_sample, :] = np.apply_along_axis(my_wl, 0, window, fs)
    
    return wl_features

def get_zc(emg, window_len, step_len, thresh, fs):
    """
    使用滑动窗口方法从EMG数据中提取零交叉特征。
    
    参数:
    - emg: 一个形状为 (Nsample, Nchannel) 的二维numpy数组，每列是一个EMG数据通道。
    - window_len: 滑动窗口的长度（以秒为单位）。
    - step_len: 滑动窗口的步长（以秒为单位）。
    - thresh: 用于检测有效零交叉的阈值。
    - fs: 采样频率（Hz）。
    
    返回:
    - zc: 提取的零交叉特征，形状为 (num_windows, Nchannel)。
    """
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    Nsample, Nchannel = emg.shape

    # 计算窗口的数量
    num_windows = (Nsample - window_sample) // step_sample + 1
    zc_features = np.zeros((num_windows, Nchannel))
    
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            zc_features[fea_idx, j] = my_zc(emg_window, thresh)
        fea_idx += 1
    
    return zc_features

def get_ssc(emg, window_len, step_len, thresh, fs):
    """
    使用滑动窗口方法从EMG数据中提取斜率符号变化特征。
    
    参数:
    - emg: 一个形状为 (Nsample, Nchannel) 的二维numpy数组，每列是一个EMG数据通道。
    - window_len: 滑动窗口的长度（以秒为单位）。
    - step_len: 滑动窗口的步长（以秒为单位）。
    - thresh: 用于检测有效斜率符号变化的阈值。
    - fs: 采样频率（Hz）。
    
    返回:
    - ssc: 提取的斜率符号变化特征，形状为 (num_windows, Nchannel)。
    """
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    Nsample, Nchannel = emg.shape

    # 计算窗口的数量
    num_windows = (Nsample - window_sample) // step_sample + 1
    ssc_features = np.zeros((num_windows, Nchannel))
    
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            ssc_features[fea_idx, j] = my_ssc(emg_window, thresh)
        fea_idx += 1
    
    return ssc_features
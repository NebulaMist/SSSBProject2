import numpy as np
from sklearn.decomposition import PCA

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
def my_zc(sig, thresh):
    """
    计算零交叉数量。
    
    参数：
    - sig: 输入的信号（1D数组）。
    - thresh: 检测有效零交叉的阈值。
    
    返回：
    - zc_value: 零交叉的数量。
    """
    N = len(sig)
    zc_value = 0
    for i in range(N - 1):
        if abs(sig[i + 1] - sig[i]) > thresh and sig[i] * sig[i + 1] < 0:
            zc_value += 1
    return zc_value


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

def my_ssc(sig, thresh):
    """
    计算斜率符号变化数量。
    
    参数：
    - sig: 输入的信号（1D数组）。
    - thresh: 检测有效斜率符号变化的阈值。
    
    返回：
    - ssc_value: 斜率符号变化的数量。
    """
    N = len(sig)
    ssc_value = 0
    for i in range(1, N - 1):
        if (sig[i] - sig[i - 1]) * (sig[i] - sig[i + 1]) > 0 and (
            abs(sig[i + 1] - sig[i]) > thresh or abs(sig[i - 1] - sig[i]) > thresh):
            ssc_value += 1
    return ssc_value



def feature_normalize(feature_train, feature_test, pca_active, dim):
    """
    归一化特征并执行PCA（如果pca_active==1）以降低维度。

    参数:
    - feature_train: numpy数组的列表，每个元素是训练试验的特征矩阵。
    - feature_test: numpy数组的列表，每个元素是测试试验的特征矩阵。
    - pca_active: 如果为1，执行PCA以降低维度。
    - dim: 如果应用PCA，则保留的主成分数量。

    返回:
    - feature_train_norm: 训练试验的归一化特征矩阵列表。
    - feature_test_norm: 测试试验的归一化特征矩阵列表。
    """
    # 确保feature_train和feature_test是数组的列表
    if not isinstance(feature_train, list):
        feature_train = [feature_train]
    if not isinstance(feature_test, list):
        feature_test = [feature_test]

    # 连接所有训练特征以计算均值和标准差
    feature_train_concat = np.hstack(feature_train)
    
    mean_val = np.mean(feature_train_concat, axis=1, keepdims=True)
    std_val = np.std(feature_train_concat, axis=1, keepdims=True)

    # 标准化训练和测试集的特征矩阵
    feature_train_norm = []
    feature_test_norm = []
    
    if pca_active == 1:
        # 在PCA之前标准化特征
        feature_train_concat = (feature_train_concat - mean_val) / std_val
        pca = PCA(n_components=dim)
        pca.fit(feature_train_concat.T)  # 对训练数据拟合PCA

    # 归一化并可选地使用PCA降低训练数据的维度
    for train_features in feature_train:
        norm_train = (train_features - mean_val) / std_val
        if pca_active == 1:
            norm_train = np.dot(norm_train.T, pca.components_[:dim].T).T  # 应用PCA转换
        feature_train_norm.append(norm_train)

    # 归一化并可选地使用PCA降低测试数据的维度
    for test_features in feature_test:
        norm_test = (test_features - mean_val) / std_val
        if pca_active == 1:
            norm_test = np.dot(norm_test.T, pca.components_[:dim].T).T  # 应用PCA转换
        feature_test_norm.append(norm_test)

    # 如果只有一个测试/训练样本，返回numpy数组而不是列表
    if len(feature_test_norm) == 1:
        feature_test_norm = feature_test_norm[0]
    if len(feature_train_norm) == 1:
        feature_train_norm = feature_train_norm[0]

    return feature_train_norm, feature_test_norm

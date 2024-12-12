import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from LDA_feature_extractor import (
    get_rms,
    get_wl,
    get_zc,
    get_ssc,
    feature_normalize
)

# 加载数据
data = sio.loadmat("")
labels = sio.loadmat("")

data = data['preprocessed_pr_motion']
label = labels['label_motion'].flatten()  # 假设它是一个列向量

task = 'dynamic'  # 如果需要，可以改为 'maintenance'

# 设置任务特定参数
if task == 'dynamic':
    window_len = 0.75  # 0.75 秒滑动窗口提取特征
    step_len = 0.75
elif task == 'maintenance':
    window_len = 3.75
    step_len = 3.75

zc_ssc_thresh = 0.0004  # 零交叉和斜率符号变化特征的阈值
fs_emg = 2048  # 采样频率
pca_active = 1  # 是否应用 PCA
dim = 200  # PCA 主成分数量

# 派生参数
Nsample = int(np.ceil(window_len * fs_emg))
Ntrial = len(data)
features = []

# 循环遍历试验以提取特征
for j in range(Ntrial):
    emg = data[0][j][-Nsample:]  # 选择最后的 Nsample 样本
    rms_tmp = get_rms(emg, window_len, step_len, fs_emg)
    wl_tmp = get_wl(emg, window_len, step_len, fs_emg)
    zc_tmp = get_zc(emg, window_len, step_len, zc_ssc_thresh, fs_emg)
    ssc_tmp = get_ssc(emg, window_len, step_len, zc_ssc_thresh, fs_emg)
    
    rms = rms_tmp.reshape(-1)
    wl = wl_tmp.reshape(-1)
    zc = zc_tmp.reshape(-1)
    ssc = ssc_tmp.reshape(-1)
    
    feature = np.concatenate([rms, wl, zc, ssc])
    features.append(feature)

features = np.array(features).T  # 形状: (features, Ntrial)

# 交叉验证以分类数据
predict_label = np.zeros(Ntrial)
for j in range(Ntrial):
    feature_test = features[:, j]
    feature_train = np.delete(features, j, axis=1)
    
    label_test = label[j]
    label_train = np.delete(label, j)
    
    # 归一化特征
    feature_train_norm, feature_test_norm = feature_normalize(feature_train, feature_test, pca_active,dim)
    
    # 训练 LDA 模型
    lda = LDA()
    lda.fit(feature_train_norm.T, label_train)
    
    # 预测测试样本的标签
    predict_label[j] = lda.predict(feature_test_norm.T)
    
# 计算准确率
accuracy = accuracy_score(label, predict_label)
print(f"Accuracy: {accuracy}")


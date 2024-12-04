import os
from scipy.io import loadmat
from LDA_feature_extractor import extract_all_features
from RF_feature_extractor import extract_all_features

# 定义根目录路径
root_dir = r''


# 定义一个函数，用于加载一个子文件夹的数据
def load_subject_data(subject_folder):
    subject_data = {
        'labels': {},
        'preprocessed': {},
        'raw': {}
    }
    subject_path = os.path.join(root_dir, subject_folder)

    # 遍历文件夹中的所有 .mat 文件
    for file_name in os.listdir(subject_path):
        file_path = os.path.join(subject_path, file_name)

        # 加载标签文件
        if 'gesture_wrong_task' in file_name:
            subject_data['labels']['gesture_wrong_task'] = loadmat(file_path)
        elif 'label_gesture' in file_name:
            subject_data['labels']['label_gesture'] = loadmat(file_path)
        elif 'label_motion' in file_name:
            subject_data['labels']['label_motion'] = loadmat(file_path)
        elif 'motion_wrong_task' in file_name:
            subject_data['labels']['motion_wrong_task'] = loadmat(file_path)

        # 加载预处理数据文件
        elif 'preprocessed_pr_gesture' in file_name:
            subject_data['preprocessed']['preprocessed_pr_gesture'] = loadmat(file_path)
        elif 'preprocessed_pr_motion' in file_name:
            subject_data['preprocessed']['preprocessed_pr_motion'] = loadmat(file_path)

        # 加载原始数据文件
        elif 'raw_pr_gesture' in file_name:
            subject_data['raw']['raw_pr_gesture'] = loadmat(file_path)
        elif 'raw_pr_motion' in file_name:
            subject_data['raw']['raw_pr_motion'] = loadmat(file_path)

    return subject_data
def label_and_process_data(subject_data):
    # 将标签文件标记在预处理数据上
    if 'label_gesture' in subject_data['labels'] and 'preprocessed_pr_gesture' in subject_data['preprocessed']:
        labels = subject_data['labels']['label_gesture']
        subject_data['preprocessed']['preprocessed_pr_gesture_labels'] = labels

    # 删除需要删除的数据
    if 'gesture_wrong_task' in subject_data['labels']:
        wrong_task = subject_data['labels']['gesture_wrong_task']
        indices_to_delete = wrong_task.get('indices', [])
        for key in subject_data['preprocessed']:
            data = subject_data['preprocessed'][key]
            subject_data['preprocessed'][key] = [d for i, d in enumerate(data) if i not in indices_to_delete]
        del subject_data['labels']['gesture_wrong_task']

# 按需加载和处理数据
for subject_folder in os.listdir(root_dir):
    subject_path = os.path.join(root_dir, subject_folder)

    # 检查是否是文件夹
    if os.path.isdir(subject_path):
        print(f"Loading data for subject: {subject_folder}")

        # 加载单个文件夹的数据
        subject_data = load_subject_data(subject_folder)

        # 特征提取
        features = extract_all_features(subject_data)
        print(f"{subject_folder} - Features:", features)

        # 处理数据和打标签
        label_and_process_data(subject_data)
        print(f"{subject_folder} - Processed Data:", subject_data['preprocessed'])


        # 清空数据，释放内存
        del subject_data

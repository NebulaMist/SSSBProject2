def extract_feature1(subject_data):
    # 实现特征提取逻辑1
    print("Extracting feature 1...")
    feature1 = {"feature1": 1}
    return feature1

def extract_feature2(subject_data):
    # 实现特征提取逻辑2
    print("Extracting feature 2...")
    feature2 = {"feature2": 2}
    return feature2

def extract_all_features(subject_data):
    # 调用所有特征提取函数
    features = {}
    features.update(extract_feature1(subject_data))
    features.update(extract_feature2(subject_data))
    # 可以继续添加更多特征提取函数
    return features
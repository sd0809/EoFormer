import os
import json
import random
from sklearn.model_selection import StratifiedShuffleSplit

# 有测试集
def split_sample(root: str, val_rate: float = 0.15, test_rate: float = 0.15): # 返回每个slice的path label

    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    random.seed(42)

    # 1.以 subgroup 区分 训练集验证集
    tumor_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]  # SHH, G3_G4, WNT
    
    # 排序，保证顺序一致
    tumor_class.sort()
    # 生成类别名称及对应的数字索引
    class_indices = dict((k,v) for v, k in enumerate(tumor_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    train_path = []
    train_label = []
    val_path = []
    val_label = []
    test_path = []
    test_label = []

    # 遍历每个文件夹下的文件
    for cla in tumor_class:
        cla_path = os.path.join(root, cla)
        # 包含所有sample的列表
        sample_lst = os.listdir(cla_path)
        val_sample_lst = random.sample(sample_lst, k = int(len(sample_lst) * val_rate))
        new_sample_lst = [ sample for sample in sample_lst if sample not in val_sample_lst ]
        test_sample_lst = random.sample(new_sample_lst, k = int(len(sample_lst) * test_rate))

        # 获取该类别索引
        sample_class = class_indices[cla]

        for sample in sample_lst:
            sample_path = os.path.join(cla_path, sample)
            if sample in val_sample_lst:
                val_path.append(sample_path)
                val_label.append(sample_class)
            elif sample in test_sample_lst:
                test_path.append(sample_path)
                test_label.append(sample_class)
            else:
                train_path.append(sample_path)
                train_label.append(sample_class)

    label_lst = train_label + val_label + test_label

    num_ependy = 0
    num_glio = 0
    num_medull = 0
    num_choroid = 0
    num_normal = 0

    for i in label_lst:
        if (i == 0):
            num_ependy += 1
        elif (i == 1):
            num_glio += 1
        elif (i == 2):
            num_medull += 1
        elif (i == 3):
            num_choroid += 1
        elif (i == 4):
            num_normal += 1
    
    num = num_ependy + num_medull + num_normal + num_choroid + num_glio
    print("{} samples were found in the dataset, including {} medulloblastoma samples, {} ependymoma samples, {} choroid plexus samples, {} glioma samples, and {} normal samples.\n".format(num, num_medull, num_ependy, num_choroid, num_glio, num_normal))

    return train_path, train_label, val_path, val_label, test_path, test_label


# 无测试集
def split_sample_(root: str, val_rate: float = 0.3 ): # 返回每个slice的path label

    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    random.seed(42)

    # 1.以 subgroup 区分 训练集验证集
    tumor_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]  # SHH, G3_G4, WNT
    
    # 排序，保证顺序一致
    tumor_class.sort()
    # 生成类别名称及对应的数字索引
    class_indices = dict((k,v) for v, k in enumerate(tumor_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    train_path = []
    train_label = []
    val_path = []
    val_label = []

    # 遍历每个文件夹下的文件
    for cla in tumor_class:
        cla_path = os.path.join(root, cla)
        # 包含所有sample的列表
        sample_lst = os.listdir(cla_path)
        val_sample_lst = random.sample(sample_lst, k = int(len(sample_lst) * val_rate))

        # 获取该类别索引
        sample_class = class_indices[cla]

        for sample in sample_lst:
            sample_path = os.path.join(cla_path, sample)
            if sample in val_sample_lst:
                val_path.append(sample_path)
                val_label.append(sample_class)
            else:
                train_path.append(sample_path)
                train_label.append(sample_class)

    label_lst = train_label + val_label

    num_ependy = 0
    num_glio = 0
    num_medull = 0
    num_choroid = 0
    num_normal = 0

    for i in label_lst:
        if (i == 0):
            num_choroid += 1
        elif (i == 1):
            num_ependy += 1
        elif (i == 2):
            num_glio += 1
        elif (i == 3):
            num_medull += 1
        elif (i == 4):
            num_normal += 1

    num = num_ependy + num_medull + num_normal + num_choroid + num_glio
    print("{} samples were found in the dataset, including {} medulloblastoma samples, {} ependymoma samples, {} choroid plexus samples, {} glioma samples, and {} normal samples.\n".format(num, num_medull, num_ependy, num_choroid, num_glio, num_normal))

    return train_path, train_label, val_path, val_label

# 无测试集
def get_data(root: str, modility_lst): # 返回所有sample的data path,  label

    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    random.seed(42)

    # 1.以 subgroup 区分 训练集验证集
    tumor_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]  # SHH, G3_G4, WNT
    
    # 排序，保证顺序一致
    tumor_class.sort()
    # 生成类别名称及对应的数字索引
    class_indices = dict((k,v) for v, k in enumerate(tumor_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    path_lst = []
    label_lst =[]

    # 遍历每个文件夹下的文件
    for cla in tumor_class:
        cla_path = os.path.join(root, cla)
        # 包含所有sample的列表
        sample_lst = os.listdir(cla_path)
        for sample in sample_lst:
            tag = 1
            sample_path = os.path.join(cla_path, sample)
            sample_class = class_indices[cla]
            sample_modility_lst = os.listdir(sample_path)
            for modility in modility_lst:  # 对于包含需要的modility_lst中所有的modility的sample, 添加到path中
                if modility not in sample_modility_lst:
                    tag = 0
                    break
            if tag == 1:
                path_lst.append(sample_path)
                label_lst.append(sample_class)

    num_ependy = 0
    num_glio = 0
    num_medull = 0
    num_choroid = 0
    num_normal = 0

    for i in label_lst:
        if (i == 0):
            num_choroid += 1
        elif (i == 1):
            num_ependy += 1
        elif (i == 2):
            num_glio += 1
        elif (i == 3):
            num_medull += 1

    num = num_ependy + num_medull + num_normal + num_choroid + num_glio
    print("{} samples were found in the dataset, including {} medulloblastoma samples, {} ependymoma samples, {} choroid plexus samples and {} glioma samples.\n".format(num, num_medull, num_ependy, num_choroid, num_glio))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # 训练&验证 ： 测试  =  4：1
    sss.get_n_splits(path_lst, label_lst)
    train_val_path, test_path = [], []
    train_val_label, test_label = [], []
    for train_val_idx, test_idx in sss.split(path_lst, label_lst):
        for idx in train_val_idx:
            train_val_path.append(path_lst[idx])
            train_val_label.append(label_lst[idx])
        for idx in test_idx:
            test_path.append(path_lst[idx])
            test_label.append(label_lst[idx])
    return train_val_path, train_val_label, test_path, test_label
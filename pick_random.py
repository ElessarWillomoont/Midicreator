from pathlib import Path
import shutil
import random
import math

# 假设dataset_dir已经定义并指向了包含JSON文件的目录
dataset_dir = Path("dataset/dataset_json")

# 定义训练集和验证集的目录
train_set_dir = dataset_dir / "train_set"
validation_set_dir = dataset_dir / "validation_set"

# 创建这些目录，如果它们不存在
train_set_dir.mkdir(parents=True, exist_ok=True)
validation_set_dir.mkdir(parents=True, exist_ok=True)

# 获取所有JSON文件的列表
json_files = list(dataset_dir.glob("*.json"))

# 随机打乱文件列表
random.shuffle(json_files)

# 计算95%的文件数量
num_train_files = math.floor(len(json_files) * 0.95)

# 分割文件列表为训练集和验证集
train_files = json_files[:num_train_files]
validation_files = json_files[num_train_files:]

# 移动文件到相应的目录
for file in train_files:
    shutil.move(str(file), str(train_set_dir / file.name))

for file in validation_files:
    shutil.move(str(file), str(validation_set_dir / file.name))

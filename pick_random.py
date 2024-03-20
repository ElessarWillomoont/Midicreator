from pathlib import Path
import shutil
import random
import math

PICK_RATIO = 0.05

# 定义目录
dataset_dir = Path("dataset/dataset_json")
unused_dir = dataset_dir / "unused"
train_set_dir = unused_dir / "train_set"
validation_set_dir = unused_dir / "validation_set"
eval_set_dir = unused_dir / "eval_set"

final_train_set_dir = dataset_dir / "train_set"
final_validation_set_dir = dataset_dir / "validation_set"
final_eval_set_dir = dataset_dir / "eval_set"

# 创建目录，如果它们不存在
unused_dir.mkdir(parents=True, exist_ok=True)
train_set_dir.mkdir(parents=True, exist_ok=True)
validation_set_dir.mkdir(parents=True, exist_ok=True)
eval_set_dir.mkdir(parents=True, exist_ok=True)

final_train_set_dir.mkdir(parents=True, exist_ok=True)
final_validation_set_dir.mkdir(parents=True, exist_ok=True)
final_eval_set_dir.mkdir(parents=True, exist_ok=True)

# 获取所有JSON文件的列表
json_files = list(dataset_dir.glob("*.json"))

# 随机打乱文件列表
random.shuffle(json_files)

# 定义分配比例
train_ratio, validation_ratio, eval_ratio = 0.85, 0.1, 0.05

# 分配文件到unused目录
num_train_files = math.floor(len(json_files) * train_ratio)
num_validation_files = math.floor(len(json_files) * validation_ratio)

train_files = json_files[:num_train_files]
validation_files = json_files[num_train_files:num_train_files + num_validation_files]
eval_files = json_files[num_train_files + num_validation_files:]

# 移动文件到相应的unused目录
for file in train_files:
    shutil.move(str(file), str(train_set_dir / file.name))
for file in validation_files:
    shutil.move(str(file), str(validation_set_dir / file.name))
for file in eval_files:
    shutil.move(str(file), str(eval_set_dir / file.name))

# 定义一个函数用于从unused目录选取文件
def pick_files(source_dir, target_dir, ratio):
    all_files = list(source_dir.glob("*.json"))
    num_pick_files = math.ceil(len(all_files) * ratio)
    picked_files = random.sample(all_files, num_pick_files)
    for file in picked_files:
        shutil.move(str(file), str(target_dir / file.name))

# 从unused目录选取文件到dataset_json目录
pick_files(train_set_dir, final_train_set_dir, PICK_RATIO)
pick_files(validation_set_dir, final_validation_set_dir, PICK_RATIO)
pick_files(eval_set_dir, final_eval_set_dir, PICK_RATIO)

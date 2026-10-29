# EE6483 Mini Project：猫狗图像分类与 CIFAR-10 迁移

基于 ImageNet 预训练的 MobileNetV3，完成 **猫狗二分类（Kaggle Dogs vs. Cats）**，并将同一套训练/评估方案迁移到 **CIFAR-10 十类分类**。项目覆盖数据构建、数据增强、Mixup/CutMix、知识蒸馏、类别不平衡方案、模型分析与可视化。

## 主要结果

### 猫狗二分类（验证集 2,000 张）

| 实验 | 数据 | 关键配置 | Val Acc |
|---|---|---|---|
| 基线 | base_1003 | 无增强 | 97.75% |
| 随机裁剪 | base_1003 | RRC 0.7–1.0 | 98.60% |
| 随机裁剪（更强） | base_1003 | RRC 0.5–1.0 | 98.35% |
| + ColorJitter | base_1003 | RRC 0.7 + CJ | 98.60% |
| + RandomErasing | base_1003 | RRC 0.7 + RE | 98.35% |
| + 水平翻转 | base_1003 | RHF + RRC 0.7 + RE | 98.55% |
| 全量训练集 | base_1007 | RRC 0.7 + RE | 98.75% |
| + 数据统计归一化 | base_1007 | RRC 0.7 + RE + dsnorm | 98.75% |
| + Mixup/CutMix | base_1007 | + MC + dsnorm | 98.95% |
| **MobileNetV3-Large** | base_1007 | RRC 0.7 + RE + MC + dsnorm | **99.45%** |
| Large → Small 蒸馏 | base_1007 | T=3, α=0.5 | 99.00% |

- 最佳模型：**MobileNetV3-Large（4,204,594 参数）**，99.45%；Large vs Small 配对检验 McNemar p≈0.013
- 增强搜索在 RC0.7 附近饱和；扩数据与换骨干收益更明显

### CIFAR-10 十类分类

| 模型 | 参数量 | Val Acc | Test Acc | Macro F1 |
|---|---|---|---|---|
| MobileNetV3-Large（stride 适配） | 4,214,842 | 97.32% | **96.30%** | **0.9629** |

- 首层与第一个下采样 bneck 的 stride 2→1，32×32 输入最终得到 4×4 特征图
- 逐类 precision / recall / F1 见 `outputs/cifar10_mobilenetv3_large/test_per_class_metrics.csv`

## 环境配置

- Python 3.12（conda 环境 `CatAndDog`）
- RTX 50 系列（sm_120）必须使用 CUDA 12.8 的 PyTorch wheel

```bash
conda create -n CatAndDog python=3.12 -y
conda activate CatAndDog
pip install -r requirements.txt
```

`requirements.txt` 已包含 `torch==2.11.0+cu128` / `torchvision==0.26.0+cu128` 与 `--extra-index-url`。

## 数据准备

### 猫狗数据

数据来自 Kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)（可使用同内容镜像 `biaiscience/dogs-vs-cats`），整理为 ImageFolder 结构：

```
data/
├── train/
│   ├── cat/   10,000 张
│   └── dog/   10,000 张
├── val/
│   ├── cat/    2,500 张
│   └── dog/    2,500 张
└── test/        500 张（无标签，仅用于推理）
```

构建训练子集：

```bash
# base_1003：每类 8,000 训练 / 1,000 验证
python tools/scripts/build_dataset.py --src-dir data --out-dir dataset/base_1003 \
    --n-train 8000 --n-val 1000 --seed 42

# base_1007：使用全量 train（每类 10,000），验证集复用 base_1003
python tools/scripts/build_dataset.py --src-dir data --out-dir dataset/base_1007 \
    --n-train 10000 --n-val 1000 --seed 42 --val-src-dir dataset/base_1003/val
```

### CIFAR-10 数据

```bash
curl -L -o data/cifar10.tgz https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
tar -xzf data/cifar10.tgz -C data/
python tools/scripts/prepare_cifar10_cache.py   # 生成 data/cifar10/cifar10_cache.pt
```

结构：`data/cifar10/train`（50,000）、`data/cifar10/test`（10,000），各 10 个类别文件夹。

## 目录结构

```
EE6483_Mini_Project/
├── model/                 # 模型定义与 build_model 工厂
│   ├── mobilenetv3_small.py
│   ├── mobilenetv3_large.py   # 含 CIFAR stride 适配版
│   └── efficientnet_b0.py
├── src/
│   ├── data.py            # 猫狗数据加载与增强
│   ├── train.py           # 猫狗训练 / 评估 / Mixup / 蒸馏
│   ├── evaluate.py        # checkpoint 评估（支持 flip TTA）
│   ├── cifar.py           # CIFAR-10 内存缓存与数据管线
│   ├── train_cifar.py     # CIFAR-10 训练入口
│   └── evaluate_cifar.py  # 逐类 precision / recall / F1
├── tools/scripts/         # 数据集、统计、可视化、错判样本导出
├── data/                  # 原始数据（不入库）
├── dataset/               # 生成的训练子集（不入库）
├── outputs/               # 权重、指标、曲线（不入库）
├── requirements.txt
└── CLAUDE.md              # 项目详细说明
```

## 快速开始

### 猫狗：训练最佳模型

```bash
python -m src.train \
    --data-dir dataset/base_1007 \
    --output-dir outputs/base_1007_large_re_mc_dsnorm \
    --backbone mobilenet_v3_large \
    --epochs 50 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 --workers 4 \
    --augment random_resized_crop,random_erasing \
    --mixup-alpha 0.2 --cutmix-alpha 1.0 --mix-prob 0.5 \
    --norm-mean 0.4863 0.4532 0.4155 --norm-std 0.2621 0.2557 0.2582
```

保存 `best.pth`、`last.pth`、`metrics.csv`，训练结束输出最终混淆矩阵。

### CIFAR-10：训练与评估

```bash
python -m src.train_cifar --data-dir data/cifar10 \
    --output-dir outputs/cifar10_mobilenetv3_large \
    --epochs 100 --batch-size 128 --lr 1e-3 --workers 8

python -m src.evaluate_cifar \
    --checkpoint outputs/cifar10_mobilenetv3_large/best.pth --split test
```

## 主要命令行参数

### `src.train`

| 参数 | 说明 |
|---|---|
| `--backbone` | `mobilenet_v3_small` / `mobilenet_v3_large` / `efficientnet_b0` |
| `--augment` | 逗号组合：`random_resized_crop`、`horizontal_flip`、`color_jitter`、`random_erasing` |
| `--rrc-scale MIN MAX` | 随机裁剪面积范围（默认 0.7 1.0） |
| `--hflip-p` / `--color-jitter` / `--erasing-p` / `--erasing-scale` | 对应增强参数 |
| `--mixup-alpha` / `--cutmix-alpha` / `--mix-prob` | Mixup / CutMix 配置 |
| `--norm-mean` / `--norm-std` | 覆盖 ImageNet 归一化 |
| `--eval-tta flip` | 验证时启用水平翻转 TTA |
| `--distill-checkpoint` / `--distill-temperature` / `--distill-alpha` | 知识蒸馏 |

### `src.train_cifar`

默认 `--backbone mobilenet_v3_large_cifar`，默认增强 RRC 0.8–1.0 + HFlip + RE，Mixup/CutMix（p=0.5），label smoothing 0.1。训练集 45,000 / 验证 5,000（分层，seed 42）/ 测试 10,000。

## 评估与可视化

```bash
# 二分类 checkpoint 评估（支持 TTA）
python -m src.evaluate --checkpoint outputs/base_1007_large_re_mc_dsnorm/best.pth \
    --split val --tta flip

# 导出错判样本（含置信度）
python tools/scripts/save_misclassified.py \
    --checkpoint small=outputs/base_1007_re_mixup_cutmix_dsnorm/best.pth \
    --checkpoint large=outputs/base_1007_large_re_mc_dsnorm/best.pth \
    --runs 3 --output-dir outputs/misclassified \
    --norm-mean 0.4863 0.4532 0.4155 --norm-std 0.2621 0.2557 0.2582

# 训练曲线 / 准确率柱状图 / 多版本对比
python tools/scripts/plot_curves.py --metrics outputs/base_1007_large_re_mc_dsnorm/metrics.csv \
    --output-dir outputs/base_1007_large_re_mc_dsnorm
python tools/scripts/plot_accuracy.py --cat-acc 99.50 --dog-acc 99.40
python tools/scripts/plot_comparison.py \
    --group "Base1007_MobileNetV3_small_RC0.7_RE" 98.90 98.60 \
    --group "Base1007_MobileNetV3_small_RC0.7_RE_MC_dsnorm" 99.00 98.90 \
    --output outputs/accuracy_comparison_1007_re_vs_mc_dsnorm.png

# 数据集统计
python tools/scripts/dataset_stats.py --data-dir dataset/base_1007
python tools/scripts/compute_stats.py --data-dir dataset/base_1007/train --workers 8
```

## 复现说明

- 所有命令在 `CatAndDog` 环境中执行
- 随机种子固定为 42（`cudnn.deterministic=True`），结果仍可能因 GPU / cuDNN 版本有微小差异
- 验证集仅用于模型选择；CIFAR-10 官方测试集只在最终评估使用
- `data/`、`dataset/`、`outputs/` 和模型权重不入库（见 `.gitignore`）

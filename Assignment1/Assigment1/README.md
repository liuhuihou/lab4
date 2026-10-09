# CS324 Assignment 1

学号：12411103

## 提交内容

| 文件 | 内容 |
|---|---|
| `report_12411103.pdf` | Part I、Part II、Part III 的方法、结果与分析 |
| `Assignment1_experiments.ipynb` | 已执行的实验 Notebook，包含结果表和图像 |
| `run_experiments.py` | 运行全部实验并生成结果文件 |
| `Part_1/task1_dataset.py` | 生成并划分二维高斯数据集 |
| `Part_1/perceptron.py` | 感知机前向传播与整批梯度训练 |
| `Part_2/modules.py` | Linear、ReLU、SoftMax 和 CrossEntropy 模块 |
| `Part_2/mlp_numpy.py` | NumPy 多层感知机 |
| `Part_2/train_mlp_numpy.py` | 全批量、SGD 和小批量训练入口 |
| `results/` | 实验数据、训练记录和结果图像 |

## 运行环境

- Python 3.10 或更高版本
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook

安装所需库：

```bash
python -m pip install numpy matplotlib scikit-learn jupyter
```

所有命令均应在本 README 所在目录运行。

## Part I：感知机

生成两类二维高斯数据，每类100个样本。每类使用80个样本训练、20个样本测试，标签为 `-1` 和 `+1`。

```bash
python Part_1/task1_dataset.py
```

`generate_gaussian_dataset()` 返回：

```text
X_train: (160, 2)
y_train: (160,)
X_test:  (40, 2)
y_test:  (40,)
```

感知机使用零初始化、学习率0.01和最多100个epoch。每个epoch汇总全部误分类样本的平均梯度，并进行一次参数更新。`errors_per_epoch` 保存每轮误分类数量。

## Part II：NumPy MLP

网络默认结构为 `2 -> 20 -> 2`，隐藏层使用ReLU，输出层使用Softmax，损失函数为平均交叉熵。数据由 `make_moons` 生成，共1000个样本，随机分为800个训练样本和200个测试样本。

运行全批量梯度下降：

```bash
python Part_2/train_mlp_numpy.py --batch_size 800 --output results/full_batch_standalone.json
```

默认参数为：

```text
hidden units: 20
learning rate: 0.01
epochs: 1500
evaluation frequency: 10
```

## Part III：SGD与batch size

运行随机梯度下降：

```bash
python Part_2/train_mlp_numpy.py --batch_size 1 --output results/sgd_standalone.json
```

运行小批量梯度下降：

```bash
python Part_2/train_mlp_numpy.py --batch_size 32 --output results/minibatch_standalone.json
```

实验比较的batch size为 `1、16、32、128、800`。每个epoch开始前都会打乱训练数据，最后不足一个完整batch的数据仍会参与更新。

## 运行全部实验

```bash
python run_experiments.py
```

程序会将结果写入 `results/`：

- `part1_results.json`：感知机实验记录；
- `mlp_results.json`：MLP训练曲线、最终指标和混淆矩阵；
- `metadata.json`：实验参数和运行环境；
- `*.npz`：固定的数据集与数据划分；
- `figures/*.png`：报告和Notebook使用的实验图像。

## 查看实验结果

```bash
jupyter notebook Assignment1_experiments.ipynb
```

Notebook 已保存完整输出，可以直接查看Part I、Part II和Part III的实验结果、准确率曲线、决策边界与batch size对比。

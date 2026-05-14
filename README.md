# Lyapunov NN Prototype

这是一个面向非线性系统稳定性分析的 Python 原型项目。项目的目标是：给定系统后，自动搜索候选 Lyapunov 函数，并通过采样验证检查其是否满足稳定性条件。

## 项目简介

Lyapunov 函数是判断系统稳定性的经典工具。传统方法通常依赖人工构造，尤其在非线性系统场景中，找到合适的函数形式并不容易。本项目尝试用一个可运行的原型，把“输入系统、搜索候选函数、验证结果”这条链路打通。

当前版本优先支持两类系统：

- 线性系统：通过矩阵 A 表示，输出二次型候选函数 V(x)=x^T P x
- 一个固定的二维非线性示例：用于演示神经网络候选函数的搜索过程

## 主要功能

- 根据系统类型自动选择候选函数族
- 训练候选 Lyapunov 函数参数
- 计算 V(x) 与 dV/dt
- 进行采样验证并输出指标
- 保存模型参数、训练历史和验证结果

## 目录结构

- `lyapunov_nn/main.py`：程序入口，负责训练、验证与结果保存
- `lyapunov_nn/systems.py`：系统定义与线性矩阵解析
- `lyapunov_nn/model.py`：候选 Lyapunov 函数模型
- `lyapunov_nn/losses.py`：损失函数与导数计算
- `lyapunov_nn/train.py`：训练流程与多次重启搜索
- `lyapunov_nn/verify.py`：采样验证与统计指标

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行线性系统示例

```bash
python -m lyapunov_nn.main --system-name linear --state-dim 2 --linear-a=-1,1,-1,-1 --epochs 100 --restarts 1 --output-dir runs/linear_demo
```

该命令会对线性系统

```text
dot x = A x
A = [[-1, 1], [-1, -1]]
```

进行候选 Lyapunov 函数搜索，最终输出 `metrics.json` 和 `best_model.pt`。

### 3. 运行一个固定的二维非线性示例

```bash
python -m lyapunov_nn.main --system-name stable_cubic_2d --output-dir runs/demo
```

## 输出结果

运行完成后，输出目录中通常包含：

- `metrics.json`：系统信息、候选函数参数与验证指标
- `best_model.pt`：训练得到的最优模型参数
- `history.pt`：训练过程记录

## 如何理解验证结果

重点查看以下三项指标：

- `positivity_violation_rate`：越接近 0 越好
- `derivative_violation_rate`：越接近 0 越好
- `dvdt_max`：通常希望小于 0

如果是线性系统，还可以重点看 `candidate_lyapunov.P` 是否接近正定矩阵，以及是否与理论解析形式一致。

## 课题阶段说明

这个项目目前定位为“可运行原型”。它不是严格的数学证明系统，但已经能展示自动构造 Lyapunov 函数的基本流程。

# Lyapunov NN

这是一个用于**自动构造 Lyapunov 候选函数**的 Python 原型项目。

它面向控制、自动化和相关方向的学习者或研究者。你只需要提供一个动力系统，程序就会尝试训练出一个满足 Lyapunov 条件的候选函数，并给出数值验证结果。

这个项目的定位不是“严格数学证明器”，而是一个**可直接上手的实验工具**：

- 给定系统，自动搜索候选 Lyapunov 函数
- 检查 $V(x) > 0$、$V(0) = 0$、$\dot V(x) < 0$
- 输出一个人能看懂的函数表达式
- 保存模型和验证结果，便于继续分析

## 适合谁

- 刚接触 Lyapunov 稳定性分析的学生
- 需要快速验证一个动力系统是否“看起来稳定”的老师或研究者
- 想把“手工找 Lyapunov 函数”改成“半自动搜索”的实验人员

## 这个项目做什么

你可以把它理解成三步：

1. 输入一个系统的动力学方程
2. 程序训练一个候选 Lyapunov 函数
3. 程序在采样点上验证这个函数是否满足条件，并输出结果

对于线性系统，它会倾向于输出二次型；对于非线性系统，它会训练一个神经网络候选函数，再把结果整理成一个显式近似表达式，方便阅读。

## 项目结构

- [lyapunov_nn/main.py](lyapunov_nn/main.py)：程序入口，负责训练、验证和保存结果
- [lyapunov_nn/systems.py](lyapunov_nn/systems.py)：内置系统与自定义系统加载
- [lyapunov_nn/model.py](lyapunov_nn/model.py)：Lyapunov 候选函数模型
- [lyapunov_nn/losses.py](lyapunov_nn/losses.py)：Lyapunov 约束损失
- [lyapunov_nn/train.py](lyapunov_nn/train.py)：训练流程
- [lyapunov_nn/verify.py](lyapunov_nn/verify.py)：采样验证
- [custom_3d_example.py](custom_3d_example.py)：一个可直接运行的自定义非线性系统示例

## 安装

建议先进入项目根目录，然后安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果你没有使用现成的虚拟环境，也可以先创建自己的 Python 环境，再执行上面的命令。

## 快速开始

### 1. 先跑一个线性系统

这个例子最容易成功，适合先检查环境有没有问题。

```powershell
.\.venv\Scripts\python.exe -m lyapunov_nn.main --system-name linear --state-dim 2 --linear-a=-1,1,-1,-1 --epochs 100 --restarts 1 --output-dir runs\\linear_demo
```

这个系统对应：

```text
dx/dt = A x
A = [[-1, 1],
     [-1, -1]]
```

### 2. 再跑一个内置非线性系统

```powershell
.\.venv\Scripts\python.exe -m lyapunov_nn.main --system-name stable_cubic_2d --epochs 100 --restarts 1 --output-dir runs\\nonlinear_demo
```

### 3. 跑你自己的系统

最推荐的方式是新建一个 Python 文件，写出系统动力学，然后把文件路径传给程序。

项目里已经放了一个示例： [custom_3d_example.py](custom_3d_example.py)

运行命令：

```powershell
.\.venv\Scripts\python.exe -m lyapunov_nn.main --custom-system-file custom_3d_example.py --state-dim 3 --epochs 100 --restarts 1 --output-dir runs\\custom_demo
```

## 如何自定义系统

你的系统文件里只要满足下面任意一种写法即可：

### 写法 1：直接定义 `system`

```python
from dataclasses import dataclass
import torch


@dataclass
class MySystem:
    name: str = "my_system"
    state_dim: int = 2

    def dynamics(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        x2 = x[..., 1]
        dx1 = -x1 - x2
        dx2 = x1 - x2 + x1 * x2
        return torch.stack((dx1, dx2), dim=-1)


system = MySystem()
```

### 写法 2：定义 `create_system()`

```python
def create_system():
    return MySystem()
```

程序会优先读取文件里的 `system`，如果没有，就尝试调用 `create_system()`。

## 输出结果在哪里看

运行结束后，`output-dir` 里通常会有这些文件：

- `best_model.pt`：训练得到的最优模型参数
- `history.pt`：训练过程记录
- `metrics.json`：系统信息、候选函数和验证指标
- `lyapunov_function.json`：导出的候选 Lyapunov 函数表达式

### 你最该看的指标

在 `metrics.json` 中，重点关注这几个字段：

- `positivity_violation_rate`：越接近 0 越好
- `derivative_violation_rate`：越接近 0 越好
- `origin_value`：越接近 0 越好
- `dvdt_max`：如果是稳定候选，通常希望它小于 0

如果这几个指标都表现良好，说明程序找到了一个不错的 Lyapunov 候选函数。

## 输出的函数是什么意思

对于线性系统，输出通常是精确的二次型：

```text
V(x) = x^T P x
```

对于非线性系统，程序会先训练一个神经网络候选函数，再把它整理成一个**可读的显式近似表达式**。这个表达式是为了展示和检查方便，不是手工指定的固定模板。

## 你可以把它用来做什么

- 快速尝试某个系统是否存在 Lyapunov 候选函数
- 对比不同系统、不同参数下的稳定性表现
- 给课程作业、开题报告或实验报告提供自动化结果
- 作为进一步严格证明的起点

## 局限性

这个项目是一个实验型工具，所以要注意：

- 它做的是采样验证，不是全空间的严格证明
- 对于复杂系统，结果可能依赖采样范围和训练次数
- 如果系统本身不稳定，或者平衡点不在原点，通常要先做坐标平移

## 常见问题

### 1. 为什么有时不同系统都能跑，但结果不一定很漂亮？

因为程序是在采样点上搜索候选函数，结果受训练、采样和系统难度影响。

### 2. 为什么建议先跑线性系统？

因为它最容易检查环境是否正常，也最容易和理论结果对应。

### 3. 自定义系统必须是 2 维吗？

不是。项目支持任意维度，只要你的 `state_dim` 和 `dynamics(x)` 写对即可。

### 4. 我怎么判断结果能不能用？

先看 `positivity_violation_rate` 和 `derivative_violation_rate` 是否接近 0，再看导出的 Lyapunov 表达式是否合理。

## 最后

如果你是第一次接触这个项目，建议按这个顺序试：

1. 跑线性系统
2. 跑内置非线性系统
3. 改 [custom_3d_example.py](custom_3d_example.py)
4. 再用你自己的动力系统测试

这样最容易理解“输入系统 -> 自动构造 Lyapunov 函数 -> 验证结果”这条流程。

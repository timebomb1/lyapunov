# Lyapunov NN Prototype

这是一个可运行的最小原型，核心目标是：

1. 你输入一个系统（早期版本先支持线性系统矩阵 A）
2. 程序自动搜索一个候选 Lyapunov 函数
3. 程序自动做采样验证并给出结果

如果你是第一次接触这个方向，可以先把它理解成：

- 你给系统方程
- 程序返回一个“稳定性评分函数”并告诉你它在采样上是否通过

## 你现在可以直接做什么

### 1) 进入项目目录

```bash
cd d:\zwj\lyapunov
```

### 2) 先跑最简单的线性系统例子（推荐）

下面这个命令表示系统

dx/dt = A x, 其中 A = [[-1, 0], [0, -2]]

程序会自动返回一个二次型候选函数：

V(x) = x^T P x

```bash
C:\Users\xn\miniconda3\envs\lyapunov\python.exe -m lyapunov_nn.main --system-name linear --state-dim 2 --linear-a=-1,0,0,-2 --epochs 100 --restarts 1 --output-dir runs/linear_demo
```

注意：

- 如果矩阵里有负数，请用 --linear-a=-1,0,0,-2 这种等号写法
- 这是为了避免命令行把 -1 误识别成新参数

### 3) 运行结束后重点看 metrics.json

文件里会出现：

- system.A：你输入的系统矩阵
- candidate_lyapunov.P：自动搜索出来的 Lyapunov 二次型矩阵
- verification：采样验证结果

也就是说，这个版本已经实现了“输入系统 -> 输出候选 Lyapunov 函数参数”。

### 4) 保留的固定非线性示例（可选）

如果你的终端已经激活了 lyapunov 环境：

```bash
python -m lyapunov_nn.main --system-name stable_cubic_2d --output-dir runs/demo
```

如果没有激活环境，直接用解释器绝对路径：

```bash
C:\Users\xn\miniconda3\envs\lyapunov\python.exe -m lyapunov_nn.main --system-name stable_cubic_2d --output-dir runs/demo
```

### 5) 运行结束后看这几个结果

- runs/demo/metrics.json：核心验证指标（包括系统信息与候选函数参数）
- runs/demo/training_curve.png：训练过程曲线
- runs/demo/lyapunov_landscape.png：2D 等高线和相平面可视化
- runs/demo/best_model.pt：训练得到的模型参数

## 怎么判断这次结果是否基本可用

先看 metrics.json 里的三个关键量：

- positivity_violation_rate：越接近 0 越好
- derivative_violation_rate：越接近 0 越好
- dvdt_max：通常希望小于 0（至少在大多数采样点上为负）

这三项满足得越好，说明候选函数越像一个可用的 Lyapunov 函数。

在线性系统模式下，还可以直接看：

- candidate_lyapunov.P 是否为正定（对角项一般应为正）

## 常用调参命令

先用快速小实验检查链路（线性系统）：

```bash
python -m lyapunov_nn.main --system-name linear --state-dim 2 --linear-a=-1,0,0,-2 --epochs 50 --restarts 1 --batch-size 128 --output-dir runs/quick
```

再用稍完整的配置提升结果：

```bash
python -m lyapunov_nn.main --system-name linear --state-dim 2 --linear-a=-1,0,0,-2 --epochs 400 --restarts 3 --batch-size 256 --output-dir runs/full
```

## 当前原型范围（你需要知道的边界）

- 目前优先支持线性系统输入（通过矩阵 A）
- 目前保留了一个固定 2D 非线性示例用于对照
- 目前是采样式数值验证，不是严格数学证明
- 适合做课题早期原型，不是最终论文级证明系统

## 代码结构

- lyapunov_nn/main.py：主入口（训练 + 验证 + 保存结果）
- lyapunov_nn/systems.py：示例非线性系统
- lyapunov_nn/model.py：结构化 Lyapunov 网络
- lyapunov_nn/losses.py：训练目标与约束损失
- lyapunov_nn/train.py：训练循环与多次重启搜索
- lyapunov_nn/verify.py：采样验证与统计
- lyapunov_nn/plot.py：结果可视化


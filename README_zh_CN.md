# 调度系统

这是一个简单的调度系统，支持不同的调度策略和资源管理模型。为了增强代码的可读性和维护性，代码被分为多个模块。

## 项目结构

该项目分为以下主要部分：

- **scheduling_strategies.py**: 包含各种调度策略类，每个类实现了不同的资源调度方式。
- **models.py**: 定义了数据模型，如 `Pod`、`UserWorkload` 和 `Node`，这些类代表系统中的关键实体。
- **scheduler.py**: 定义了 `Scheduler` 类，该类与不同的调度策略进行交互，执行调度操作。
- **main.py**: 程序的入口点，负责初始化必要的组件并触发调度过程。

### 文件详细说明

1. **scheduling_strategies.py**:
   - **SchedulingStrategy**: 一个抽象基类（ABC），用于定义调度策略的接口。
   - **RandomSchedulingStrategy**: 实现了随机调度策略。
   - **LeastResourceSchedulingStrategy**: 实现了最少资源调度策略，选择使用最少资源的节点。
   - **DelayAwareSchedulingStrategy**: 实现了延迟感知调度策略，考虑任务延迟来进行调度。

2. **models.py**:
   - **Pod**: 表示一个需要被调度的工作负载单元。
   - **UserWorkload**: 表示用户的工作负载，可能包含多个 `Pod` 实例。
   - **Node**: 表示系统中的一个节点，`Pod` 可以被调度到该节点上。

3. **scheduler.py**:
   - **Scheduler**: 一个类，它接收一个 `SchedulingStrategy` 并基于该策略进行资源调度。

4. **main.py**:
   - 主程序脚本，初始化调度策略、调度器实例和模型（`Pod`、`UserWorkload` 和 `Node`），并演示如何触发调度过程。

## 使用说明

1. **安装 Python**: 确保你已经安装了 Python 3。

2. **运行主程序**:
   - 进入项目目录，并执行以下命令运行程序：
     ```bash
     python main.py
     ```

3. **更改调度策略**:
   - 在 `main.py` 中，你可以修改策略初始化的代码行，切换不同的调度策略：
     ```python
     # 例如使用 LeastResourceSchedulingStrategy 而不是 RandomSchedulingStrategy
     strategy = LeastResourceSchedulingStrategy()
     ```

4. **扩展项目**:
   - 若要添加新的调度策略，请在 `scheduling_strategies.py` 中创建一个继承自 `SchedulingStrategy` 的新类，并实现 `schedule` 方法。
   - 若要引入新的模型或资源，请在 `models.py` 中添加新的类。

## 示例输出

`main.py` 脚本将根据所选的调度策略运行调度过程。输出内容将取决于具体的调度策略逻辑以及 `Pod`、`UserWorkload` 和 `Node` 实例的详细信息。

## 未来改进

- 增加更多高级调度策略。
- 实现更复杂的资源管理模型。
- 扩展 `Scheduler` 类以同时处理多个调度请求。
- 引入日志记录和错误处理机制。

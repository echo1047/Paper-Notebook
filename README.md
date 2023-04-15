# Paper-Notebook

## planning in stochastic environments with a learned model

### Stochastic Model
在推理过程中，给定初始观测值$o_{\le t}$和动作序列$a_{t:t+K}$，我们可以递归展开上述模型，并从分布$c_{t+k+1} \sim \sigma_{t}^{k}$中抽样机会结果，来生成轨迹。
### (A) 蒙特卡罗树搜索
随机MuZero中采用蒙特卡罗树搜索，其中菱形节点表示机会节点，圆形节点表示决策节点。在选择过程中，对于机会节点采用先验分布$\sigma$抽样选择边,对于决策节点采用pUCT公式选择边。
### (B) 训练随机模型
例如给出一条长度为2的轨迹,包括观测$o_{\le t:t+2}$,动作$a_{t:t+2}$,价值目标$z_{t:t+2}$,策略目标${\pi}_{t:t+2}$,奖励$u_{t+1:t+K}$. 模型将推演两步

编码器$e$输入为$o_{\le t+k}$,确定性地生产偶然结果$c_{t+k}$. 模型的策略,价值和奖励结果分别向目标逼近

对未来的编码的分布$\sigma^{k}$进行训练，以预测编码器产生的编码。

### 随机搜索
在随机MuZero中引入机会节点和机会值，扩展了MCTS算法。在MCTS的随机实例化中，存在两种节点类型:decision和chance (Couetoux, 2013)。机会节点和决策节点沿着树的深度交错排列，因此每个决策节点的父节点都是一个机会节点。树的根节点总是一个决策节点。

在我们的方法中，每个机会节点对应一个潜在的后状态(4.1)，它通过查询随机模型进行扩展，其中父状态和一个动作作为输入，模型返回节点的价值和未来编码$Pr(c|as)$上的先验分布。在一个机会节点被扩展后，它的值被反向传播到树中。最后，当在选择阶段遍历节点时，通过对先验分布采样来选择一个编码。在Stochastic MuZero中，每个内部决策节点通过查询学习到的模型再次扩展，其中机会节点的父节点的状态和一个采样的编码c作为输入，模型返回一个奖励、一个价值和一个策略。与MuZero类似，新添加节点的值向上反向传播到树中，并使用pUCT(2)公式来选择一条边。随机MuZero使用的随机搜索如图1所示。


主要的创新点:

# planning in stochastic environments with a learned model

## 4.1 Stochastic Model

在推理过程中，给定初始观测值$o_{\le t}$和动作序列$a_{t:t+K}$，我们可以递归展开上述模型，并从分布$c_{t+k+1} \sim \sigma_{t}^{k}$中抽样机会结果，来生成轨迹。

### 偶然结果 (chance outcome)

前面看到, VAE的隐变量$z$的每一维都是一个连续的值, 而VQ-VAE最大的特点就是, z的每一维都是离散的整数.
将离散化的关键就是VQ, 即vector quatization. 简单来说, 就是要先有一个codebook, 这个codebook是一个embedding table. 我们在这个table中找到和vector最接近(比如欧氏距离最近)的一个embedding, 用这个embedding的index来代表这个vector. 而commitment loss用来约束encoder，这里的$\beta$为权重系数，论文默认设置为0.25。 ​

![Alt text](images/6C8GSL5%25LG5%25~XD%60G1HKG%7B1.png)

随机MuZero通过使用VQ-VAE方法的一种新变体来模拟机会结果。具体地，我们考虑一个大小为M的常量码本的VQ-VAE。码本中的每一项都是一个大小为M的固定独热向量。利用一个独热向量的固定码本，我们可以简化VQ-VAE的方程。在这种情况下，我们将嵌入$c^e_t$的编码器建模为一个分类变量，选择最接近的编码$c_t$相当于计算表达式$\text { one hot }(\arg \max _{i}(c_{t}^{e, i}))$。由此产生的编码器也可以视为观测的随机函数，该函数利用了Gumbel softmax重参数化技巧(Jang等人，2016)，向前通过时温度为零，向后通过时为直通估计器。在我们的模型中没有显式解码器，与之前的工作相反(Ozair et al.， 2021)，我们没有使用重构损失。相反，该网络以类似于MuZero的方式进行端到端训练。在下一节中，我们将更详细地解释培训过程。

>首先从类型上来讲，encoder和decoder指的是模型，embedding指的是tensor。
>encoder 编码器，将信息/数据进行编码，或进行特征提取（可以看做更复杂的编码）
>decoder 解码器，将特征解码为任务相关的输出。如词向量->词，feature map->检测框+类别序列，feature -> 生成图像。
>embedding 比较特殊，在不同任务和模型下会有具体的指代。一般来讲，我们会对简单处理后的数字化的数据叫embedding，如transformer前的token。但是在有的场景下，也会管模型（如encoder）输出的特征叫embedding。

>总结一下：encoder 用来提取、凝练（降维）特征embedding 指由某种能理解的模态提取得到的特征或数字化的张量decoder用于将特征向量转化为我们需要的结果。

> 讲完了VQ-AVE的大致思路，我们会发现，现在学习到的又是一个固定的codebook，这也意味着它又没办法像VAE一样，通过随机采样生成图片，准确的说VQ-VAE并不像一个VAE，而更像一个AE，它学习到的codebook适用于类似分类的任务而不是生成任务，如果想要VQ-VAE做生成任务，就需要像论文里提到的一样，再训练一个prior网络，利用codebook实现图像的生成任务。

### 模型训练

$$L^{\text {total }}=L^{\text {MuZero }}+L^{\text {chance }}$$

$$L^{\text {MuZero }}=\sum_{k=0}^{K} l^{p}\left(\pi_{t+k}, p_{t}^{k}\right)+\sum_{k=0}^{K} l^{v}\left(z_{t+k}, v_{t}^{k}\right)+\sum_{k=1}^{K} l^{r}\left(u_{t+k}, r_{t}^{k}\right)$$

其中,策略目标$\pi_{t+k}$为从观测值$o_{\le t+k}$搜索得到的MCTS策略. 价值目标$z_{t+k}$用多步回报计算, 奖励目标$u_{t+k}$为观测到的即时回报.

$$L_{w}^{\text {chance }}=\sum_{k=0}^{K-1} l^{Q}\left(z_{t+k}, Q_{t}^{k}\right)+\sum_{k=0}^{K-1} l^{\sigma}\left(c_{t+k+1}, \sigma_{t}^{k}\right)+\beta \sum_{k=0}^{K-1}\left\|c_{t+k+1}-c_{t+k+1}^{e}\right\|^{2}$$

前两项与muzero loss保持一致,第三项用VQ-VAE commitment cost来确保编码器的输出$c_{t+k}^{e}=e(o_{\leq t+k})$逼近编码$c_{t+k}$

$c_{t+k+1}=\operatorname{one} \operatorname{hot}\left(\arg \max _{i}\left(e\left(o_{\leq t+k+1}^{i}\right)\right)\right)$

### (A) 蒙特卡罗树搜索

随机MuZero中采用蒙特卡罗树搜索，其中菱形节点表示机会节点，圆形节点表示决策节点。在选择过程中，对于机会节点采用先验分布$\sigma$抽样选择边,对于决策节点采用pUCT公式选择边。

### (B) 训练随机模型

例如给出一条长度为2的轨迹,包括观测$o_{\le t:t+2}$,动作$a_{t:t+2}$,价值目标$z_{t:t+2}$,策略目标${\pi}_{t:t+2}$,奖励$u_{t+1:t+K}$. 模型将推演两步

编码器$e$输入为$o_{\le t+k}$,确定性地生产偶然结果$c_{t+k}$. 模型的策略,价值和奖励结果分别向目标逼近

对未来的编码的分布$\sigma^{k}$进行训练，以预测编码器产生的编码。

> e和h一样是对观测结果的抽象, 取前n帧观测作为输入,具体取决于不同的游戏
### 随机搜索

在随机MuZero中引入机会节点和机会值，扩展了MCTS算法。在MCTS的随机实例化中，存在两种节点类型:decision和chance (Couetoux, 2013)。机会节点和决策节点沿着树的深度交错排列，因此每个决策节点的父节点都是一个机会节点。树的根节点总是一个决策节点。

在我们的方法中，每个机会节点对应一个潜在的后状态(4.1)，它通过查询随机模型进行扩展，其中父状态和一个动作作为输入，模型返回节点的价值和未来编码$Pr(c|as)$上的先验分布。在一个机会节点被扩展后，它的值被反向传播到树中。最后，当在选择阶段遍历节点时，通过对先验分布采样来选择一个编码。在Stochastic MuZero中，每个内部决策节点通过查询学习到的模型再次扩展，其中机会节点的父节点的状态和一个采样的编码c作为输入，模型返回一个奖励、一个价值和一个策略。与MuZero类似，新添加节点的值向上反向传播到树中，并使用pUCT(2)公式来选择一条边。随机MuZero使用的随机搜索如图1所示。


---
前置工作是AlphaZero，AlphaZero采用了启发式搜索（MCTS）+强化学习+自博弈的方法，在围棋领域取得了很好的效果，但是类似AlphaZero这种规划算法基于环境动力学已知，需要先验游戏规则或精确模拟器。Muzero的贡献在AlphaZero强大的搜索和策略迭代算法的基础上加入了模型学习的过程，使其能够在不了解状态转移规则的情况下，达到了当时的SOTA效果。

最传统的MBRL工作一般是建立transition model和reward model，而MBRL的另一条线是价值等价模型，这个思想最早由ICML2017的The predictron: End-to-end learning and planning这篇文章提出。这类方法的主要思想是构建一个抽象的MDP模型，从相同的真实状态出发，在抽象MDP中rollout轨迹的累积奖励与真实环境中的轨迹累积奖励相匹配，这种模型最大的特点是不要求transition model能够拟合真实状态，仅需要预测V。

最后是一些细节性的设置，比如围棋中用的是连续的8个棋盘状态，Atari输入的是96$*$96的连续32个RGB帧。围棋用的动作维度是19$*$19的one-hot。


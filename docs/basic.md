## 多任务学习基础

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-some-notation-in-MTL.png)

在这里给出任务的定义：**变量概率分布**（反映为数据）$p_{i}(\mathbf{x})$，**条件概率分布**（反映为模型）$p_{i}(\mathbf{y} | \mathbf{x})$ 和**损失函数** $\mathscr{L}_{i}$ 共同构成一个**任务**。$p_{i}(\mathbf{x})$ 是真实数据的生成分布，实际上无法访问，但是每个任务中的训练集和测试集间接反映了分布。

多任务分类中，不同任务共用同一个损失函数；而多标签学习中，不同任务共用相同的数据和损失函数。什么时候不同任务会使用不同的损失函数？(1) 一些任务的标签是离散的，另一些任务的标签是连续的； (2) 对多任务中的某些任务更关心。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-task-descriptor.png)

对于具体任务，我们可以将 $f_{\theta}(\mathbf{y} | \mathbf{x})$ 改写为 $f_{\theta}(\mathbf{y} | \mathbf{x},\mathbf{z}_i )$。其中，$\mathbf{z}_i$ 是任务描述符，可以是任务编号的 one-hot 编码，也可以是任务的任何元数据，例如任务的语言描述或者正式说明。另外，我们将目标函数从单任务变为多任务。模型和算法需要考虑的有两点：(1) 如何将任务描述符 $\mathbf{z}_i$ 作为条件；(2) 如何优化目标。

### 任务描述符 $\mathbf{z}_i$ 作为条件

如果我们想用任务描述符 $\mathbf{z}_i$ 来控制，让任务之间尽可能少地分享参数，则可以让每个任务拥有一个独立的神经网络，最后用 $$\mathbf{z}_i$$ 来决定输出哪一个网络的预测值。这样，任务之间没有共享参数。另一个极端是，我们可以将 $\mathbf{z}_i$ 与输入或激活值拼接，这样能够使除了紧接在 $\mathbf{z}_i$ 后的参数（可能有的矩阵变换等）以外的所有参数都能够共享。

因此，我们可以将模型的参数 $\theta$ 划分为共享参数 $\theta^{sh}$ 和任务特定参数 $\theta^{i}$。则目标函数变为

$$
\min _{\theta^{s h}, \theta^{1}, \ldots, \theta^{T}} \sum_{i=1}^{T} {L}_{i}\left(\left\{\theta^{s h}, \theta^{i}\right\}, D_{i}\right)
$$

这样，如何将任务描述符 $\mathbf{z}_i$ 作为条件的问题转变为设计如何共享参数，以及在哪共享参数。一些常见的设计选择如下图所示。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-conditioning-some-common-choices-1.png)

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-conditioning-some-common-choices-2.png)

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-conditioning-more-complex-choices.png)

然而，这些设计选择基本上都与神经网络架构调整相对应，因此，它们

- 是问题特定的。为解决一个问题的设计在另外的一个或一组问题上不一定能很好地工作；
- 很大程度上由直觉或者对问题地了解来指导；
- 比起科学，更像是艺术，没有指南。

### 优化目标

一种针对多任务学习的基础优化方案如下：

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-MTL-optimizing-basic-version.png)

### 挑战

#### 负迁移

有时训练独立的网络比多任务学习方法能获得更好的效果，这意味着**负迁移（negative transfer）**的发生。负迁移指在某些任务上学到的知识，对于其他任务的学习产生负面效果。负迁移的产生原因可能是优化带来的挑战，例如在优化某一任务的梯度时可能会使得其他任务的梯度优化更加困难（可能类似于陷入局部最优），或者任务学习的速度不同，某一任务的快速学成也使得其他任务在开始学习前已经学到了一些不想学到的知识；另一种原因是网络的表达能力可能有限，因此多任务学习的网络通常比单任务网络大得多。

如果发生了负迁移，可以在任务间减少共享信息或者加大网络规模。注意，共享参数不是简单的“共享”或“不共享”的非正即负的选择，也可以采取软参数共享（soft parameter sharing）。例如，用一个正则化项来鼓励任务特定的参数彼此相似。这样的做法可以可以通过一个平衡因子来调整参数共享的程度，从而使得参数共享的自由度更高。缺点是引入了额外的设计部分（决定在哪里采取软参数共享）和超参数。

#### 过拟合

如果每个任务只有少量数据，并且发生过拟合，可能是因为这些任务之间互相分享的信息太少。增加数据量自然是一种缓解过拟合的方法，你也可以选择共享更多信息。


## 元学习基础

我们可以从两种角度来看元学习算法，机械的角度有助于我们实现元学习算法，而概率论的角度能够帮助我们更好地理解元学习算法。

从机械的角度来看，有一种神经网络模型能够读取整个数据集并为新数据点做出预测。我们使用元数据集来训练这个网络，这个元数据集包含很多数据集，每个数据集用于一个任务。

从概率论的角度来看，元学习是在（元训练阶段）从一系列任务中提取先验知识，来更有效地学习新任务。学习新任务时，我们使用先验知识和（较少的）训练数据来推导出最有可能的后验参数。

### 问题定义

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-meta-problem-definitions-1.png)

我们可以将监督学习看作最大似然问题，即在给定数据的情况下最大程度地提高参数的可能性。通过贝叶斯公式，可以将其重新定义为在给定参数的情况下最大化数据的概率，并最大化参数的边缘概率。前者可以看作数据的可能性，而后者可以看作是权重衰减等正则化器。当数据量较少时，模型可能会对数据过拟合（即使有正则化器）。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-meta-problem-definitions-2.png)

当我们解决监督学习问题时，我们会希望不是从头开始学起，而是已经在附加的元训练数据上获得一定的经验，使得在新任务上能够更有效地学习。这里的元训练数据指一组与任务相关的任务或数据集。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-meta-learning-problem.png)

考虑到每次有新任务时都访问元训练数据的耗费较多，我们希望从元训练数据上学到一组元参数（meta-parameters）$\theta$ 来加速新任务的学习，而非保留元训练数据。之后，可以通过**适应**（**adaptation**）来从 $\theta$ 得到新任务的参数 $\phi$。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-a-quick-example.png)


元学习的基本准则是测试和训练的条件应保持一致。优化 $\theta$ 的过程是元训练，而为产生 $\phi$ 而进行的优化过程是元测试。简而言之，通常的机器学习流程中的训练和测试阶段在元学习范式下变为元训练和元测试阶段，而通常训练和测试阶段中的数据点在元学习范式下变为单个任务，其中每个任务都有自己的训练集和测试集。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-the-complete-meta-learning-optimization.png)

上图是一个完整的元学习优化流程。简单来说，元学习的目的就是学到一个足够好的 $\theta$ 使得经过适应新任务的训练数据 $D^{tr}_i$ 得到的 $\phi$ 对于新任务的测试数据 $D^{ts}_i$ 足够好。如图右下角所示，我们也可以将其看作一个图模型，其中 $\theta$ 本质上是先验（跨任务共享的信息），$\phi_i$ 是给定任务的任务特定参数。注意我们不知道元测试阶段测试集的标签。

为了更好地和通常的机器学习设置区分，我们习惯将元学习中每个任务中的训练集和测试集称为**支持集（support set）**和**查询集（query set）**。另外的一个名词是 k-shot 学习，这里的“k-shot”是指每个任务中支持集中每个类别的样本个数为 k。

### 与其他问题的共性与区别

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-closely-related-problem-settings.png)

元学习的目的是通过一系列任务学到一组参数，这组参数能够通过适应新任务的训练数据来得到一组对新任务最好的任务特定参数。多任务学习的目的是得到能在当前的一组任务上表现优秀的参数，而不在乎在新任务上的表现。

从元学习的角度来看，超参数优化中的 $\theta$ 是超参数，$\phi$ 是网络的权重；AutoML 中的 $\theta$ 是架构，$\phi$ 是网络权重。

## 参考资料

* 本节内容对应的 [PPT](http://web.stanford.edu/class/cs330/slides/cs330_lec2.pdf)
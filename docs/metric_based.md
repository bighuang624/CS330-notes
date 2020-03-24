## 非参数小样本学习

迄今为止我们一直在学习有参数的模型，然而非参数方法（例如，最近邻分类器）在低数据环境下简单易用且表现良好。考虑到在元测试阶段，小样本学习正是在低数据环境下进行的，因此这些非参数方法可能可以执行得很好。当然，在元训练期间，我们仍希望模型是参数化的，以扩展到大型数据集。因此，这类方法的关键思想在于使用有参数的元学习器来产生有效的非参数学习器。

一种非常自然的想法是，对于一张测试图像，我们可以将它与所有训练图像进行对比，来找到一张最相似的训练图像，并返回这张训练图像所对应的标签。这就是最近邻分类。这样做的问题在于我们应该使用什么距离度量来进行这个对比过程？在元学习过程中，我们可以学习如何来进行对比过程，或者说，学习这个距离度量。

### Siamese Network

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-Siamese-Network.png)

ICML 2015 的论文 "[Siamese Neural Networks for One-Shot Image Recognition](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf)" 提出使用 Siamese network（孪生网络）来判断两张图像是否属于同一个类。在元训练阶段，我们训练好这个网络后，在元测试阶段就可以将每张测试图像与其所在任务的训练数据中的每张图像进行对比，然后输出与最接近的训练图像（即网络输出的可能性值最高的训练图像）所对应的标签。

在元训练阶段，我们在做二元分类；而在元测试阶段，我们在做 N 类别分类。一个问题在于，我们能否和之前介绍的方法一样，让元训练和元测试阶段需要执行的程序一致？

### Matching Network

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-Matching-Network.png)

根据以上讨论，NeurIPS 2016 的论文 "[Matching Networks for One Shot Learning](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf)" 提出 Matching network。Matching network 通过两个编码器（在 Matching network 中，这两个编码器可以共享参数，但一般不共享）来分别将训练图像和测试图像投射到嵌入空间，之后进行对比来对每张训练图像的标签产生一个权重，最后得到加权和来代表对测试图像的预测。在原论文中，对应于训练图像的编码器使用了一个双向的 LSTM，而对应于测试图像的编码器使用了卷积神经网络。整个模型端到端训练，并且元训练和元测试阶段需要执行的程序是一致的，都是在将每张测试图像与其所在任务的训练数据中的每张图像进行对比，来实现 N 类别分类。

### 其他非参数方法

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-non-parametric-methods.png)

如上图所示，非参数方法的通用程序可以表示为：

1. 采样一个或一批任务 $T_{i}$；
2. 将每个任务切分为不相交的训练集（或者说支持集）$\mathcal{D}_{i}^{\mathrm{tr}}$ 和测试集（或者说查询集）$\mathcal{D}_{i}^{\mathrm{ts}}$；
3. 计算 $\hat{y}^{\mathrm{ts}}=\sum_{x_{k}, y_{k} \in \mathcal{D}^{\mathrm{tr}}} f_{\theta}\left(x^{\mathrm{ts}}, x_{k}\right) y_{k}$（与其他方法不同，这里没有任务特定参数 $\phi$ 的存在，因此这些方法被称为非参数方法）；
4. 使用 $\nabla_{\theta} \mathcal{L}\left(\hat{y}^{\mathrm{ts}}, y^{\mathrm{ts}}\right)$ 来更新 $\theta$。重复以上过程直到收敛。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-Prototypical-Network.png)

Matching network 存在的一个问题是，如果在同一个任务中每个类别有多个样本，Matching network 需要对每个样本分别执行与测试样本的对比，这样即加大了运算量，也可能提高了离群点对算法准确率的损害。因此，NeurIPS 2017 的论文 "[Prototypical Networks for Few-shot Learning](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf)" 提出了 Prototypical network（原型网络），在元测试阶段将每个类别的训练样本的嵌入进行平均来得到这个类别的原型（prototype），每个测试样本同样被投射到同一嵌入空间并基于被选定的距离度量（论文提出当选用平均值作为类别原型时，欧式距离是最佳的距离度量选择）进行最近邻分类。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-other-non-parametric-methods.png)

前文中提到的方法基本可以归类为将数据点投射到一个学习得到的嵌入空间，并在嵌入空间中执行最近邻分类。一个挑战是如何对数据点之间更加复杂的关系进行推理，而不仅仅是在嵌入空间中执行最近邻分类。理论上来说，如果编码器有足够强的表达能力，那么在嵌入空间中的最近邻应该能够代表各种复杂的关系（尤其是对于高维嵌入空间）。但是在实践中，用更加具有表达能力的方法来执行比较被发现有更好的效果。基于这样的思想，有以下工作被提出：

- CVPR 2018 的论文 "[Learning to Compare: Relation Network for Few-Shot Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf)"：提出 Relation network（关系网络），基本上采用 Prototypical network 来得到各种嵌入，在此基础上学习训练和测试样本间的非线性关系来取代预先设定的欧氏距离或其他距离度量。即，整个网络同时学习嵌入空间和距离度量。

- ICML 2019 的论文 "[Infinite Mixture Prototypes for Few-Shot Learning](https://arxiv.org/pdf/1902.04552.pdf)"：提出每个类别用一组簇来表示（而非像 Prototypical network 一样每个类别用单个簇来表示），即让嵌入空间具有多峰分布。

- ICLR 2018 的论文 "[Few-Shot Learning with Graph Neural Networks](https://openreview.net/forum?id=BJj6qGbRW)"：用图神经网络和消息传递算法来建模数据点之间的关系，从而帮助预测。

## 元学习算法的对比

我们可以从两种角度来比较目前学过的三大类元学习算法（黑箱适应方法、基于优化的算法、无参数方法）。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-Comparison-from-computation-graph-perspective.png)

从计算图的角度来看，黑箱适应方法以完全黑箱的方式来表示用 $D_i^{tr}$ 和 $x^{ts}$ 得到 $y^{ts}$ 的计算图；基于优化的算法可以看作将梯度优化过程嵌入到了该计算图中。而无参数方法也可看作是将某一部分设置嵌入到了计算图中，例如最近邻。因此，我们也可以考虑来对计算图中的部件进行混合和匹配，例如：

- 2019 年的论文 "[CAML: Fast Context Adaptation via Meta-Learning](https://openreview.net/forum?id=BylBfnRqFm)"：将参数分为两部分，一部分参数称为上下文参数，作为模型的附加输入可以对单个任务进行自适应；一部分参数称为共享参数，通过元训练得到并在任务间共享。尽管本文认为自己的方法效果很好，Finn 指出在实践中这种处理可能是多余的。

- ICLR 2019 的论文 "[Meta-Learning with Latent Embedding Optimization](https://openreview.net/forum?id=BJgklhAcK7)"：本文使用了 Relation network 将模型参数从高维空间编码到一个低维的潜在空间，并在该潜在空间中执行基于优化的元学习方法，这样相比优化整个模型的参数更加高效。之后再用解码器来得到进行预测所需的参数。

- ICLR 2020 的论文 "[Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples](https://openreview.net/forum?id=rkgAGAVKPr)"：在方法上，本文首先论证 Prototypical network 可以看作在特征提取器提取的特征上的线性分类器，之后为了结合 Prototypical network 具有较好的归纳偏置和 MAML 有对新任务的适应过程的优点，提出将 MAML 算法采用的模型的最后一个线性层进行约束使其实现与 Prototypical network 相同的作用。

从算法属性的角度来看，我们主要关心各元学习算法的表达能力和一致性。表达能力很重要的原因是它代表了算法对于更多更大的数据集的可扩展性，以及是否能适用于一系列领域。而一致性指算法学习一个一致的学习过程（例如梯度下降）能够解决有足够数据的任务（而无论数据是什么）。一致性很重要的原因是它可以帮助减少对元训练任务的依赖，并在分布之外的任务上取得较好表现。我们由此获得了下图所示的对于算法的比较：

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/CS330-Comparison-from-algorithmic-properties-perspective.png)

除开上述两个算法属性，还有一个较为重要的属性是对不确定性的认识，以及在学习过程中对歧义进行推理的能力。这个属性比较重要的原因是我们会想通过主动学习（active learning）、经过校准的不确定性估计，或者在强化学习设置下，知道收集哪些数据可以减少对任务的不确定性。

## 参考资料

* 本节内容对应的 [PPT](http://web.stanford.edu/class/cs330/slides/cs330_lecture4.pdf)




# 用因果树和因果森林计算异质处理效应

**Gehuiming Zhu**    

May 2022



## 异质处理效应（HTE）

$$
\begin{aligned}
X & \sim \Lambda \\
W & \sim \operatorname{Bern}(e(X)) \\
Y(0) &=\mu_{0}(X)+\varepsilon(0) \\
Y(1) &=\mu_{1}(X)+\varepsilon(1)
\end{aligned}
$$

传统的因果推断分析，主要是在平均意义上展开的，其关注的焦点是平均处理效应（average treatment effect）：
$$
\mathrm{ATE}:=\mathbb{E}[Y(1)-Y(0)]
$$
越来越多的学者开始关注处理效应的异质性。比如说，大量的以政策分析为导向的研究关注特定人群之间有差异的处理效应；假设处理方法是一种药物，该药物的人群平均效应可能不是阳性的，但是对特定类别的患者可能有效，





估计个体处理效应时的数据由（*Xi*，*Wi*，*Yi*）3部分组成，其中*i*＝1，2，…，*n*代表个体，*Xi*为一组协变量向量，*Wi*为处理分配变量，*Wi*∈{0，1}表示2种不同的处理，*Yi*为结局变量。个体处理效应的公式可定义为：
$$
\tau(x):=\mathbb{E}[D \mid X=x]=\mathbb{E}[Y(1)-Y(0) \mid X=x],
$$

> **Condition 1:**
> $$
> (\varepsilon(0), \varepsilon(1)) \perp W \mid X .
> $$
> **Condition 2:** There exists $e_{\min }$ and $e_{\max }$, such that for all $x$ in the support of $X$,
> $$
> 0<e_{\min }<e(x)<e_{\max }<1 .
> $$



## Why Machine Learning for CATE estimation?

> “传统的回归模型通过交互项来分析处理效应异质性([Aiken et al.,1991](javascript:void(0);))。之后方法论的发展则日渐依托于倾向值(propensity score)的估算，将处理效应异质性问题转为考察处理效应如何随着个体倾向值的变化而变化([Xie & Wu, 2005](javascript:void(0););[Xie et al.,2012](javascript:void(0););[Carneiroet al.,2010](javascript:void(0););[吴晓刚，2008](javascript:void(0);))。这些分析方法虽然展示了处理效应异质性估计的多种策略，但各有其不足之处。随着机器学习方法与社会科学因果推断分析的日渐结合，一个前沿的方法论发展方向是使用基于算法的技术手段来考察处理效应异质性。” “当我们有足够的计算资源来针对数据使用比较复杂的算法时，我们则不得不正视算法模型在社会科学领域内可能扮演的重要角色。这方面，因果推断技术与机器学习算法的结合正是当下社会科学方法论发展的前沿方向。”   
>
> ——胡安宁 吴晓刚 陈云松 《[处理效应异质性分析——机器学习方法带来的机遇与挑战](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=SHXJ202101005&uniplatform=NZKPT&v=AlM_7h2Ff4ylnmx_UGCdsCu4wL2jrWRgp4D7EZfZDvPNEp6b5SLLc47U-hu_pz1n)》







目前估计个体处理效应的方法有贝叶斯自适应回归树[[1](http://html.rhhz.net/zhlxbx/20190620.htm#b1)]、反事实随机森林[[2](http://html.rhhz.net/zhlxbx/20190620.htm#b2)]等。2015年，Athey和Imbens[[3](http://html.rhhz.net/zhlxbx/20190620.htm#b3)]将机器学习中常用的分类回归树（classification and regression trees）引入到了传统的因果识别框架，定义了因果树（causal tree）的概念，用它们来考察异质性处理效应。而后Wager和Athey[[4](http://html.rhhz.net/zhlxbx/20190620.htm#b4)]又推广了因果树方法，讨论了如何用随机森林（random forest）算法来整合因果树并估计异质性处理效应，称为因果森林（causal forests）。





## 由决策树谈起

决策树是一个预测模型，它代表的是对象属性与对象值之间的一种映射关系。它是一树状结构，它的每一个叶节点对应着一个分类，非叶节点对应着在某个属性上的划分，根据样本在该属性上的不同取值将其划分成若干个子集。



<img src="https://s2.loli.net/2022/05/23/wY7LHg8ZmSVN4uG.png" alt="截屏2022-05-23 08.59.56" style="zoom:50%;" />

有了这样一棵决策树，我们就很容易判断一个瓜是好瓜还是坏瓜。然而我们只有这样的数据集：

<img src="https://s2.loli.net/2022/05/23/tMz8HqA1yOIfK4U.png" alt="截屏2022-05-23 09.04.56" style="zoom:50%;" />

从数据产生决策树的机器学习技术叫做决策树学习，通俗说就是决策树。典型的决策树算法包括ID3、C4.5、CART。ID3中使用了信息增益选择特征，增益大优先选择。C4.5中，采用信息增益率选择特征，减少因特征值多导致信息增益大的问题。



## CART型决策树

ID3和C4.5算法，生成的决策树是多叉树，只能处理分类不能处理回归。而CART（classification and regression tree）分类回归树算法，既可用于分类也可用于回归。 分类树的输出是样本的类别， 回归树的输出是一个实数。

### CART分类型决策树

CART分类树算法使用 “基尼指数” (Gini index)来选择划分属性，基尼系数代表了模型的不纯度，基尼系数越小，不纯度越低，特征越好。这和信息增益（率）相反。数据集 $D$ 的纯度可用基尼值来度量:
$$
\begin{aligned}
\operatorname{Gini}(D) &=\sum_{k=1}^{|\mathcal{Y}|} \sum_{k^{\prime} \neq k} p_{k} p_{k^{\prime}} \\
&=1-\sum_{k=1}^{|\mathcal{Y}|} p_{k}^{2} .
\end{aligned}
$$
直观来说, $\operatorname{Gini}(D)$ 反映了从数据集 $D$ 中随机抽取两个样本, 其类别标记 不一致的概率. 因此, $\operatorname{Gini}(D)$ 越小, 则数据集 $D$ 的纯度越高.属性 $a$ 的基尼指数定义为
$$
\text { Gini index }(D, a)=\sum_{v=1}^{V} \frac{\left|D^{v}\right|}{|D|} \operatorname{Gini}\left(D^{v}\right) \text {. }
$$
于是, 我们在候选属性集合 $A$ 中, 选择那个使得划分后基尼指数最小的属性作为最优划分属性, 即
$$
a_{*}=\underset{a \in A}{\arg \min } \operatorname{Gini} \operatorname{index}(D, a)
$$
**Example from [Zhihu](https://zhuanlan.zhihu.com/p/383539744):**

<img src="https://s2.loli.net/2022/05/23/xI2rTcBAg8vJimX.png" alt="截屏2022-05-23 09.46.53" style="zoom:40%;" />



数据如上表所示，分别以 ![[公式]](https://www.zhihu.com/equation?tex=A_%7B1%7D%2CA_%7B2%7D%2CA_%7B3%7D%2CA_%7B4%7D) 表示属性年龄、工作、房子、信贷这4个特征，并以1，2，3表示年龄中的青年，中年，老年，以1，2表示工作和房子中的是和否，以1，2，3表示信贷中的非常好、好、一般，求各特征的基尼指数，选择最优切分点：

首先计算 $A_{1}$ 的基尼指数, 我们知道CART是以二分类树来进行构建的, 所以我们在计算任何一个 属性的基尼指数时, 都需要把计算的分为一类, 其他的作为另一类, 由此可知, 在计算 $A_{1}$ 时, 也就是把“青年”看作一类, “中年”和“老年”看作另一类, 即先要计算“青年”的可得：

$$
\operatorname{Gini}\left(D, A_{1}=1\right)=\frac{5}{15}\left(2 \times \frac{2}{5} \times\left(1-\frac{2}{5}\right)\right)+\frac{10}{15}\left(2 \times \frac{7}{10} \times\left(1-\frac{7}{10}\right)\right)=0.44
$$
其中的 $A_{1}=1$ 表示“青年”的 Gini指数 ,$\frac{5}{15}$ 表示青年样本占总样本的比值, 2 服从伯努利分布的计算 $2 p(1-p), \frac{2}{5}$ 表示青年样本中“是”类别的数量与青年样本数比值, $1-\frac{2}{5}$ 表示 “否”类别的数量与青年样本数比值, “+”后面的则代表“中年” 和“老年”。 根据以上的计算规则, 可以得到
$$
\begin{aligned}
&\operatorname{Gini}\left(D, A_{1}=2\right)=0.48 \\
&\operatorname{Gini}\left(D, A_{1}=3\right)=0.44
\end{aligned}
$$
由于 $\operatorname{Gini}\left(D, A_{1}=3\right)$ 和 $\operatorname{Gini}\left(D, A_{1}=1\right)$ 相等且最小, 所以 $A_{1}=1, A_{1}=3$ 都可以作为选择 $A_{1}$ 的最优切分点。同理, 可求得特征 $A_{2}$ 和 $A_{3}$ 的基尼指数:
$$
\begin{aligned}
&\operatorname{Gini}\left(D, A_{2}=1\right)=0.32 \\
&\operatorname{Gini}\left(D, A_{3}=1\right)=0.27
\end{aligned}
$$
由于 $A_{2}, A_{3}$ 只有一个切分点Q, 所以它们为最优切分点。最后求得 $A_{4}$ 的基尼指数：
$$
\begin{aligned}
&\operatorname{Gini}\left(D, A_{4}=1\right)=0.36 \\
&\operatorname{Gini}\left(D, A_{4}=2\right)=0.47 \\
&\operatorname{Gini}\left(D, A_{4}=3\right)=0.32
\end{aligned}
$$
所以对于 $A_{4}$ 来说, $\operatorname{Gini}\left(D, A_{4}=3\right)$ 为最优切分点。

最后进行综合考虑, 在 $A_{1}, A_{2}, A_{3}, A_{4}$ 这4个属性中, $A_{3}=1$ 为 $0.27$ 最小, 所以成为最优的切分点。于是根节点生成两个子节点, 一个是叶节点。对另一个子节点继续按照上述方法进行计算 Gini指数, 再选择一个最优切分点, 以此类推, 可以构建一棵完整的树。在本例中, 由于后续计算的最优节点为 $A_{2}=1$, 所以所得的节点都是叶节点。



### CART回归型决策树

假设X和Y分别为输入和输出变量，并且Y是连续变量，给定训练数据集 ![[公式]](https://www.zhihu.com/equation?tex=D%3D%5Cleft%5C%7B%28x_%7B1%7D%2Cy_%7B1%7D%29%2C%28x_%7B2%7D%2Cy_%7B2%7D%29%2C...%2C%28x_%7BN%7D%2Cy_%7BN%7D%29+%5Cright%5C%7D)。考虑如何生成回归树。

<img src="/Users/zhugehuiming/Library/Application Support/typora-user-images/截屏2022-05-11 14.21.28.png" alt="截屏2022-05-11 14.21.28" style="zoom:50%" />



**step1：**寻找最优切分变量j和最优切分点s的方法为

<img src="https://pic2.zhimg.com/80/v2-e847d072f6eced57dd54ae30322bfd79_1440w.png" alt="img" style="zoom:80%;" />



​          其中，![[公式]](https://www.zhihu.com/equation?tex=%7Bc%7D_%7B1%7D%3Dave%28y_%7Bi%7D%7Cx_%7Bi%7D+%5Cin+R_%7B1%7D%28j%2Cs%29%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=%7Bc%7D_%7B2%7D%3Dave%28y_%7Bi%7D%7Cx_%7Bi%7D+%5Cin+R_%7B2%7D%28j%2Cs%29%29)



**step2：**用选定的（j，s）划分区域，并决定输出值。

​             两个划分的区域分别是： ![[公式]](https://www.zhihu.com/equation?tex=R_%7B1%7D+%3D+%5Cleft%5C%7B+1%2C+2%2C+3%2C+4%2C+5+%5Cright%5C%7D%2CR_%7B2%7D+%3D+%5Cleft%5C%7B+6%2C+7%2C+8%2C+9%2C+10+%5Cright%5C%7D) 。输出值用公式：

​            ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bc%7D_%7B1%7D%3Dave%28y_%7Bi%7D%7Cx_%7Bi%7D+%5Cin+R_%7B1%7D%28j%2Cs%29%29)和 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bc%7D_%7B2%7D%3Dave%28y_%7Bi%7D%7Cx_%7Bi%7D+%5Cin+R_%7B2%7D%28j%2Cs%29%29)。得到 ![[公式]](https://www.zhihu.com/equation?tex=c_%7B1%7D%3D+5.06%2C+c_%7B2%7D%3D8.18)。



**step3：**对两个子区域继续调用算法流程中的step1、2。



<img src="https://pic3.zhimg.com/80/v2-057995da1f00bd2cfad6945995511b16_1440w.jpg" alt="img" style="zoom:50%;" />





**如何评价我们的回归树切割得好不好？**







## 随机森林







##  因果树

一些公式的详细推导过程参照：*Susan Athey and Guido Imbensa*，**《Recursive partitioning for heterogeneous causal effects》**

### 相较于普通回归型决策树的改进

#### 1.MSE公式的修改

原来的 ![[公式]](https://www.zhihu.com/equation?tex=MSE%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%5Cleft%5C%7B%28Y_i-%5Chat%7BY%7D%29%5E2%5Cright%5C%7D) ;修改后的 ![[公式]](https://www.zhihu.com/equation?tex=MSE%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%5Cleft%5C%7B%28Y_i-%5Chat%7BY%7D%29%5E2-Y_i%5E2%5Cright%5C%7D)





#### 2.Honset Approach

将原来的训练集样本随机分成两部分，一部分用于生成树（train set），另一部分用于给出预测值（estimate set）。







#### 3.直接估计异质处理效应









<img src="/Users/zhugehuiming/Library/Application Support/typora-user-images/截屏2022-05-11 10.35.19.png" alt="截屏2022-05-11 10.35.19" style="zoom:60%;" />

相较于决策树，每棵因果树split的分裂准则修改如下：


$$
\begin{aligned}
-\mathrm{EMSE}_{\mu}\left(S^{t r}, N^{e s t}, \Pi\right)=& \underbrace{\frac{1}{N^{t r}} \sum_{i \in S^{t r}} \hat{\tau}^{2}\left(x_{i} \mid S^{t r}, \Pi\right)}_{\text {Variance of treatment effect across leaves }} \\
&-\underbrace{\left(\frac{1}{N^{t r}}+\frac{1}{N^{e s t}}\right) \sum_{\ell \in \Pi}\left(\frac{S_{S_{\text {treat }}^{t r}}^{2}(\ell)}{p}+\frac{S_{S_{\text {control }}^{t r}}^{2}(\ell)}{1-p}\right)}_{\text {Uncertainty about leaf treatment effects }}
\end{aligned}
$$

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+%5Chat%7B%5Ctau%7D%28x%3BS%5E%7Btr%7D%2C%5Cprod%29%26%3D%5Chat%7B%5Cmu%7D%281%2Cx%3BS%5E%7Btr%7D%2C%5Cprod%29-%5Chat%7B%5Cmu%7D%280%2Cx%3BS%5E%7Btr%7D%2C%5Cprod%29++%5Ctag%7B5%7D%5C%5C+%5Chat%7B%5Cmu%7D%28w%2Cx%3BS%5E%7Btr%7D%2C%5Cprod%29%26%3D%5Cbar%7BY%7D_%7Bi%3Ai%5Cin+S%5E%7Btr%7D_w%2CX_i%5Cin+l%28x%7C%5Cprod%29%7D+%5Ctag%7B6%7D+%5Cend%7Balign%7D+)








## 因果森林











## Appendix: R for CATE estimation with Casual Tree and Casual Forest

**This example comes from the tutorial of Prof. Susan Athey, Stanford University*

```R
library(tidyverse)
library(tidyselect)
library(dplyr)       # Data manipulation (0.8.0.1)
library(fBasics)     # Summary statistics (3042.89)
library(corrplot)    # Correlations (0.84)
library(psych)       # Correlation p-values (1.8.12)
library(grf)         # Generalized random forests (0.10.2)
library(rpart)       # Classification and regression trees, or CART (4.1-13)
library(rpart.plot)  # Plotting trees (3.0.6)
library(treeClust)   # Predicting leaf position for causal trees (1.1-7)
library(car)         # linear hypothesis testing for causal tree (3.0-2)
library(devtools)    # Install packages from github (2.0.1)
library(readr)       # Reading csv files (1.3.1)
library(tidyr)       # Database operations (0.8.3)
library(tibble)      # Modern alternative to data frames (2.1.1)
library(knitr)       # RMarkdown (1.21)
library(kableExtra)  # Prettier RMarkdown (1.0.1)
library(ggplot2)     # general plotting tool (3.1.0)
library(haven)       # read stata files (2.0.0)
library(aod)         # hypothesis testing (1.3.1)
library(evtree)      # evolutionary learning of globally optimal trees (1.0-7)
library(purrr)
```


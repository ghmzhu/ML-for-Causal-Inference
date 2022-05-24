# 用因果树计算异质处理效应

**Gehuiming Zhu**    

May 2022



## 1 异质处理效应（HTE）

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



## 2 Why Machine Learning for CATE estimation?

> “同样是拟合一个线性回归模型，传统的社会科学研究者将注意力放在这个模型中特定自变量的系数上，而机器学习的目的则是看这个回归模型多大程度上可以预测因变量的取值。这种关注点上的区分非常重要。因为我们在进行模型拟合时所需要特别关注的问题（例如共线性等）在机器学习的分析范式下便不再是问题。只要有助于提升预测的准确度，我们的模型拟合过程完全可以变得非常有弹性。”
>
> ——胡安宁，《中国社会科学报》



> 传统回归模型的交互项分析主要有两个问题：交互项不能无限制添加，以及交互项的设定具有主观性。倾向值方法存在模型和系数不确定，以及无法确定具体起异质化作用的混淆变量的问题。
>
> ——胡安宁,吴晓刚,陈云松, [《处理效应异质性分析——机器学习方法带来的机遇与挑战》]



目前估计个体处理效应的方法有贝叶斯自适应回归树[[1](http://html.rhhz.net/zhlxbx/20190620.htm#b1)]、反事实随机森林[[2](http://html.rhhz.net/zhlxbx/20190620.htm#b2)]等。2015年，Athey和Imbens[[3](http://html.rhhz.net/zhlxbx/20190620.htm#b3)]将机器学习中常用的分类回归树（classification and regression trees）引入到了传统的因果识别框架，定义了因果树（causal tree）的概念，用它们来考察异质性处理效应。而后Wager和Athey[[4](http://html.rhhz.net/zhlxbx/20190620.htm#b4)]又推广了因果树方法，讨论了如何用随机森林（random forest）算法来整合因果树并估计异质性处理效应，称为因果森林（causal forests）。



## 3 由决策树谈起

决策树是一个预测模型，它代表的是对象属性与对象值之间的一种映射关系。它是一树状结构，它的每一个叶节点对应着一个分类，非叶节点对应着在某个属性上的划分，根据样本在该属性上的不同取值将其划分成若干个子集。



<img src="https://s2.loli.net/2022/05/23/wY7LHg8ZmSVN4uG.png" alt="截屏2022-05-23 08.59.56" style="zoom:50%;" />

有了这样一棵决策树，我们就很容易判断一个瓜是好瓜还是坏瓜。然而我们只有这样的数据集：

<img src="https://s2.loli.net/2022/05/23/tMz8HqA1yOIfK4U.png" alt="截屏2022-05-23 09.04.56" style="zoom:50%;" />

从数据产生决策树的机器学习技术叫做决策树学习，通俗说就是决策树。典型的决策树算法包括ID3、C4.5、CART。ID3中使用了信息增益选择特征，增益大优先选择。C4.5中，采用信息增益率选择特征，减少因特征值多导致信息增益大的问题。

我们将所有样本随机分为训练集和测试集，在训练集上训练出一颗决策树，并放在测试集上评价效果。

<img src="https://s2.loli.net/2022/05/23/gACsDY57bjS68U3.png" alt="截屏2022-05-23 20.33.25" style="zoom:50%;" />



## 4 CART型决策树

ID3和C4.5算法，生成的决策树是多叉树，只能处理分类不能处理回归。而CART（classification and regression tree）分类回归树算法，既可用于分类也可用于回归。 分类树的输出是样本的类别， 回归树的输出是一个实数。

### 4.1 CART分类型决策树

CART分类树算法使用 “基尼指数” (Gini index)来选择划分属性，基尼系数代表了模型的不纯度，基尼系数越小，不纯度越低，特征越好。这和信息增益（率）相反。数据集 $D$ 的纯度可用基尼值来度量:
$$
\begin{aligned}
\operatorname{Gini}(D) &=\sum_{k=1}^{|\mathcal{Y}|} \sum_{k^{\prime} \neq k} p_{k} p_{k^{\prime}} \\
&=1-\sum_{k=1}^{|\mathcal{Y}|} p_{k}^{2} .
\end{aligned}
$$
直观来说, $\operatorname{Gini}(D)$ 反映了从数据集 $D$ 中随机抽取两个样本, 其类别标记不一致的概率. 因此, $\operatorname{Gini}(D)$ 越小, 则数据集 $D$ 的纯度越高.属性 $a$ 的基尼指数定义为
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

![img](https://pic2.zhimg.com/v2-28c8f0bc0394ca8ea9b4ec0924a65fbd_b.jpg)



### 4.2 CART回归型决策树

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

每棵决策树在第i次split的时候，分裂准则如下（这里关注回归树）：

$$
\arg \min _{\prod_{i}} M S E_{i}=\frac{1}{n} \sum_{i=1}^{n}\left(Y_{i}-\bar{Y}_{j: x_{j} \in l\left(x_{i} \mid \prod_{i}\right)}\right)^2
$$
其中 $l\left(x_{i} \mid \prod_{i}\right)_{\text {表示在 }} \prod_{i}$ 的划分情况下, $x_{i}$ 所在的叶子结点。

即使在当前树在分裂完毕后，我们也可以基于交叉验证的思路，通过改变训练集测试集的划分，得到不同的树，再相互比较MSE，取到一颗最优的树。



### 4.3 剪枝







### 4.4 随机森林

**下面是随机森林的构造过程：**

1. 假如有N个样本，则有放回的随机选择N个样本(每次随机选择一个样本，然后返回继续选择)。这选择好了的N个样本用来训练一个决策树，作为决策树根节点处的样本。

2. 当每个样本有M个属性时，在决策树的每个节点需要分裂时，随机从这M个属性中选取出m个属性，满足条件m << M。然后从这m个属性中采用某种策略（比如说信息增益）来选择1个属性作为该节点的分裂属性。

3. 决策树形成过程中每个节点都要按照步骤2来分裂（很容易理解，如果下一次该节点选出来的那一个属性是刚刚其父节点分裂时用过的属性，则该节点已经达到了叶子节点，无须继续分裂了）。一直到不能够再分裂为止。注意整个决策树形成过程中没有进行剪枝。

4. 按照步骤1~3建立大量的决策树，这样就构成了随机森林了。
5. 最后对这些决策树采用投票或计算平均值的方法得到最终的结果。



##  5 因果树

**Casual Tree:** Athey S, Imbens G. [Recursive partitioning for heterogeneous causal effects](https://www.pnas.org/doi/abs/10.1073/pnas.1510489113)[J]. Proceedings of the National Academy of Sciences, 2016, 113(27): 7353-7360.

**Casual Forests:** Wager S, Athey S. [Estimation and inference of heterogeneous treatment effects using random forests](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1319839?journalCode=uasa20)[J]. Journal of the American Statistical Association, 2018, 113(523): 1228-1242.

### 5.1 相较于普通回归型决策树的改进

**1.Honset Approach**

将原来的训练集样本随机分成两部分，一部分用于生成树（train set），另一部分用于给出预测值（estimate set）。即在训练样本上训练模型，模型训练好以后放到**估计样本**上计算估计值，最后使用该估计值在测试集上计算MSE来判断模型的好坏。

**2.修改了MSE的表达式，更聚焦于Treatment Effect。**

普通的CART回归型决策树分裂以及评价的标准都是看MSE。具体来说：
$$
\arg \min _{\prod_{i}} M S E_{i}=\frac{1}{n} \sum_{i=1}^{n}\left(Y_{i}-\bar{Y}_{j: x_{j} \in l\left(x_{i} \mid \prod_{i}\right)}\right)
$$
其中 $l\left(x_{i} \mid \prod_{i}\right)$表示在 $ \prod_{i}$ 的划分情况下, $x_{i}$ 所在的叶子结点。

因果树对上述MSE的公式做了改进，主要有两点变化：一是直接用$\tau$代替了MSE公式中的$Y$；二是减去了$\tau^2$。
$$
\operatorname{MSE}_{\tau}\left(\mathcal{S}^{\mathrm{te}}, \mathcal{S}^{\text {est }}, \Pi\right) \equiv \frac{1}{\#\left(\mathcal{S}^{\mathrm{te}}\right)} \sum_{i \in \mathcal{S}^{\mathrm{te}}}\left\{\left(\tau_{i}-\hat{\tau}\left(X_{i} ; \mathcal{S}^{\text {est }}, \Pi\right)\right)^{2}-\tau_{i}^{2}\right\}
$$
通过对上述公式进行一系列的变形处理（一些公式的详细推导过程参照：*Susan Athey and Guido Imbensa*，**《Recursive partitioning for heterogeneous causal effects》**），改写成：


$$
\begin{aligned}
-\mathrm{EMSE}_{\mu}\left(S^{t r}, N^{e s t}, \Pi\right)=& \underbrace{\frac{1}{N^{t r}} \sum_{i \in S^{t r}} \hat{\tau}^{2}\left(x_{i} \mid S^{t r}, \Pi\right)}_{\text {Variance of treatment effect across leaves }} \\
&-\underbrace{\left(\frac{1}{N^{t r}}+\frac{1}{N^{e s t}}\right) \sum_{\ell \in \Pi}\left(\frac{S_{S_{\text {treat }}^{t r}}^{2}(\ell)}{p}+\frac{S_{S_{\text {control }}^{t r}}^{2}(\ell)}{1-p}\right)}_{\text {Uncertainty about leaf treatment effects }}
\end{aligned}
$$

$$
\begin{aligned}
\hat{\tau}\left(x ; S^{t r}, \prod\right) &=\hat{\mu}\left(1, x ; S^{t r}, \prod\right)-\hat{\mu}\left(0, x ; S^{t r}, \prod\right) \\
\hat{\mu}\left(w, x ; S^{t r}, \prod\right) &=\bar{Y}_{i: i \in S_{w}^{t r}, X_{i} \in l(x \mid \Pi)}
\end{aligned}
$$



### 5.2 因果森林

因果森林实现步骤如下:

采用无放回抽样从原始数据集 $\{1, \cdots, N\}$ 中随 机抽取样本量为 $s(s<N$, 默认比例为 50%) 的子集 $b$, 继而将其随机分成样本量为 $s / 2$ 的两等份, 分别作为样本 $\mathrm{T}$ 和样本 $\mathrm{E}$ 。上述过程中涉及两个样本即 $\mathrm{T}$ 和 $\mathrm{E}$, 样本 $\mathrm{T}$ 用于节点分割的选择,样本E用于个体处理效应 $\hat{\tau}_{i}(x)$ 的计算, 从而使因果树具有 “诚实(honest)”的性质, 在 因果树的构建过程中, 节点分割准则是基于 $\hat{\tau}_{i}(x)$ $(i \in \mathrm{T})$ 的方差最大化。

基于递归分区的方式生成一棵因果树,即从根 节点开始自顶向下对样本进行划分, 基于 $X_{i} \leqslant x$ 或 $X_{i}>x(i \in \mathrm{T})$ 按照节点分割准则将父节点分裂为左右 两个子节点, 然后子节点按照相同的准则继续分割, 直到新的节点不再生成为止。一棵因果树生成后, 利用公式计算每个叶子节点上个体的处理效应$\hat{\tau}_{i}(x)$。

重复上述步骤 B 次, 最终形成具有 B 棵树的因果森林, 此时第 $i$ 个个体的处理效应综合 $\mathrm{B}$ 棵树的均值进行计算, 公式为:
$$
\hat{\tau}_{i}(x)=\frac{1}{B} \sum_{b=1}^{B} \hat{\tau}_{i, b}(x)
$$



## 6 Example of Causal Trees in R

`*This is an example from the tutorial of Prof. Susan Athey, Stanford University`

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

```R
install_github('susanathey/causalTree') # Uncomment this to install the causalTree package
library(causalTree)
```



**Loading the data**

```R
source('load_data.R') 
df_experiment <- select_dataset("welfare")
```



**Cleaning the data**

```R
# Combine all names
all_variables_names <- c(outcome_variable_name, treatment_variable_name, covariate_names)
df <- df_experiment %>% select(all_variables_names)

# Drop rows containing missing values
df <- df %>% drop_na()

# Rename variables
df <- df %>% rename(Y=outcome_variable_name,W=treatment_variable_name)

# Converting all columns to numerical and add row id
df <- data.frame(lapply(df, function(x) as.numeric(as.character(x))))
df <- df %>% mutate_if(is.character,as.numeric)
df <- df %>% rowid_to_column( "ID")
                        
# Use train_fraction % of the dataset to train our models                        
train_fraction <- 0.80  
df_train <- sample_frac(df, replace=F, size=train_fraction)
df_test <- anti_join(df,df_train, by = "ID")#need to check on larger datasets                       
```



**Descriptive statistics**

```R
# Make a data.frame containing summary statistics of interest
summ_stats <- fBasics::basicStats(df)
summ_stats <- as.data.frame(t(summ_stats))
# Rename some of the columns for convenience
summ_stats <- summ_stats %>% select("Mean", "Stdev", "Minimum", "1. Quartile", "Median",  "3. Quartile", "Maximum")
summ_stats <- summ_stats %>% rename('Lower quartile'= '1. Quartile', 'Upper quartile' ='3. Quartile')
```

<img src="https://s2.loli.net/2022/05/24/5mZExdyPUwsA1ID.png" alt="截屏2022-05-24 09.13.49" style="zoom:50%;" />



 **Split the dataset**

```R
# Diving the data 40%-40%-20% into splitting, estimation and validation samples
split_size <- floor(nrow(df_train) * 0.5)
df_split <- sample_n(df_train, replace=FALSE, size=split_size)

# Make the splits
df_est <- anti_join(df_train,df_split, by ="ID")
```



**Fit the tree**

```R
fmla_ct <- paste("factor(Y) ~", paste(covariate_names, collapse = " + "))
print('This is our regression model')
print( fmla_ct)
```

```
[1] "factor(Y) ~ hrs1 + partyid + income + rincome + wrkstat + wrkslf + age + polviews + educ + earnrs + race + wrkslf + marital + sibs + childs + occ80 + prestg80 + indus80 + res16 + reg16 + mobile16 + family16 + parborn + maeduc + degree + sex + race + born + hompop + babies + preteen + teens + adults"
```

```R
ct_unpruned <- honest.causalTree(
  formula = fmla_ct,            # Define the model
  data = df_split,              # Subset used to create tree structure
  est_data = df_est,            # Which data set to use to estimate effects

  treatment = df_split$W,       # Splitting sample treatment variable
  est_treatment = df_est$W,     # Estimation sample treatment variable

  split.Rule = "CT",            # Define the splitting option
  cv.option = "TOT",            # Cross validation options
  cp = 0,                       # Complexity parameter

  split.Honest = TRUE,          # Use honesty when splitting
  cv.Honest = TRUE,             # Use honesty when performing cross-validation

  minsize = 10,                 # Min. number of treatment and control cases in each leaf
  HonestSampleSize = nrow(df_est)) # Num obs used in estimation after building the tree

rpart.plot(
  x = ct_unpruned,        # Pruned tree
  type = 3,             # Draw separate split labels for the left and right directions
  fallen = TRUE,        # Position the leaf nodes at the bottom of the graph
  leaf.round = 1,       # Rounding of the corners of the leaf node boxes
  extra = 100,          # Display the percentage of observations in the node
  branch = 0.1,          # Shape of the branch lines
  box.palette = "RdBu") # Palette for coloring the node
```

![plot_zoom](https://s2.loli.net/2022/05/24/1RUWPCuLB3sOXzp.png)

**Cross-validate**

```r
# Table of cross-validated values by tuning parameter.
ct_cptable <- as.data.frame(ct_unpruned$cptable)

# Obtain optimal complexity parameter to prune tree.
selected_cp <- which.min(ct_cptable$xerror)
optim_cp_ct <- ct_cptable[selected_cp, "CP"]

# Prune the tree at optimal complexity parameter.
ct_pruned <- prune(tree = ct_unpruned, cp = optim_cp_ct)

rpart.plot(
  x = ct_pruned,        # Pruned tree
  type = 3,             # Draw separate split labels for the left and right directions
  fallen = TRUE,        # Position the leaf nodes at the bottom of the graph
  leaf.round = 1,       # Rounding of the corners of the leaf node boxes
  extra = 100,          # Display the percentage of observations in the node
  branch = 0.1,          # Shape of the branch lines
  box.palette = "RdBu") # Palette for coloring the node
```

<img src="https://s2.loli.net/2022/05/24/GgdUF2Peo7fxYTX.png" alt="截屏2022-05-24 09.55.21" style="zoom:50%;" />








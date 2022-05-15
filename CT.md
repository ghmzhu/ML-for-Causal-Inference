# 用因果树和因果森林计算异质处理效应

**Gehuiming Zhu**    

May 2022





## 异质处理效应(CATE)

传统的因果推断





## Why Machine Learning for CATE estimation?

> “传统的回归模型通过交互项来分析处理效应异质性([Aiken et al.,1991](javascript:void(0);))。之后方法论的发展则日渐依托于倾向值(propensity score)的估算，将处理效应异质性问题转为考察处理效应如何随着个体倾向值的变化而变化([Xie & Wu, 2005](javascript:void(0););[Xie et al.,2012](javascript:void(0););[Carneiroet al.,2010](javascript:void(0););[吴晓刚，2008](javascript:void(0);))。这些分析方法虽然展示了处理效应异质性估计的多种策略，但各有其不足之处。随着机器学习方法与社会科学因果推断分析的日渐结合，一个前沿的方法论发展方向是使用基于算法的技术手段来考察处理效应异质性。” “当我们有足够的计算资源来针对数据使用比较复杂的算法时，我们则不得不正视算法模型在社会科学领域内可能扮演的重要角色。这方面，因果推断技术与机器学习算法的结合正是当下社会科学方法论发展的前沿方向。”   
>
> ——胡安宁 吴晓刚 陈云松 《[处理效应异质性分析——机器学习方法带来的机遇与挑战](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=SHXJ202101005&uniplatform=NZKPT&v=AlM_7h2Ff4ylnmx_UGCdsCu4wL2jrWRgp4D7EZfZDvPNEp6b5SLLc47U-hu_pz1n)》



## 由决策树谈起

决策树是一种机器学习的方法，它是一种树形结构（可以是二叉树或者非二叉树），其中每个内部节点表示一个属性上的判断，每个分支代表一个判断结果的输出，最后每个叶节点代表一种分类结果。







## CART回归型决策树

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


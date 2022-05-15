# An interpretation of《Metalearners for estimating heterogeneous treatment effects using machine learning》

***Gehuiming Zhu, Jiahao Zhang, Jiaxiong Wang***

May, 2022



## 1 Brief Introduction

### 1.1 Why Machine Learning?

> “同样是拟合一个线性回归模型，传统的社会科学研究者将注意力放在这个模型中特定自变量的系数上，而机器学习的目的则是看这个回归模型多大程度上可以预测因变量的取值。这种关注点上的区分非常重要。因为我们在进行模型拟合时所需要特别关注的问题（例如共线性等）在机器学习的分析范式下便不再是问题。只要有助于提升预测的准确度，我们的模型拟合过程完全可以变得非常有弹性。”
>
> “所谓的‘因果推断的基本问题’，本质上是一个缺失值问题，而为了解决缺失值问题，我们就需要利用已有的资料进行某种意义上的“预测”，这恰恰是机器学习方法的强项所在。”
>
> ——胡安宁，《中国社会科学报》



> 传统回归模型的交互项分析主要有两个问题：交互项不能无限制添加，以及交互项的设定具有主观性。倾向值方法存在模型和系数不确定，以及无法确定具体起异质化作用的混淆变量的问题。
>
> ——胡安宁,吴晓刚,陈云松, [《处理效应异质性分析——机器学习方法带来的机遇与挑战》](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=SHXJ202101005&uniplatform=NZKPT&v=AlM_7h2Ff4ylnmx_UGCdsCu4wL2jrWRgp4D7EZfZDvNWsybwUMoF6Rjdygqh_jWv)



**Casual Tree:** Athey S, Imbens G. [Recursive partitioning for heterogeneous causal effects](https://www.pnas.org/doi/abs/10.1073/pnas.1510489113)[J]. Proceedings of the National Academy of Sciences, 2016, 113(27): 7353-7360.

**Casual Forests: **Wager S, Athey S. [Estimation and inference of heterogeneous treatment effects using random forests](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1319839?journalCode=uasa20)[J]. Journal of the American Statistical Association, 2018, 113(523): 1228-1242.

**BART:** Chipman H A , Mcculloch G R E . [BART: BAYESIAN ADDITIVE REGRESSION TREES](https://arxiv.org/pdf/0806.3286.pdf)[J]. Annals of Applied Statistics, 2010, 4(1):266-298.

**Transfer Learning:** Künzel S R, Stadie B C, Vemuri N, et al. [Transfer learning for estimating causal effects using neural networks](https://arxiv.org/abs/1808.07804)[J]. arXiv preprint arXiv:1808.07804, 2018.



### 1.2 Framework and Definitions

- We use the following representation of $\mathcal{P}$ :
  $$
  \begin{aligned}
  X & \sim \Lambda \\
  W & \sim \operatorname{Bern}(e(X)) \\
  Y(0) &=\mu_{0}(X)+\varepsilon(0) \\
  Y(1) &=\mu_{1}(X)+\varepsilon(1)
  \end{aligned}
  $$

-  $\left(Y_{i}(0), Y_{i}(1), X_{i}, W_{i}\right) \sim \mathcal{P}$, where $X_{i} \in \mathbb{R}^{d}$ is a $d$-dimensional covariate or feature vector, $W_{i} \in\{0,1\}$ is the treatment-assignment indicator (to be defined precisely later), $Y_{i}(0) \in \mathbb{R}$ is the potential outcome of unit $i$ when $i$ is assigned to the control group, and $Y_{i}(1)$ is the potential outcome when $i$ is assigned to the treatment group. With this definition, the `ATE` is defined as

$$
\mathrm{ATE}:=\mathbb{E}[Y(1)-Y(0)]
$$
- `the response under control` $\mu_{0}(x):=\mathbb{E}[Y(0) \mid X=x] \quad$ 
- `the response under treatment`$\quad \mu_{1}(x):=\mathbb{E}[Y(1) \mid X=x]$

- For a new unit $i$ with covariate vector $x_{i}$, to decide whether to give the unit the treatment, we wish to estimate the `ITE` of unit $i, D_{i}$, which is defined as

$$
D_{i}:=Y_{i}(1)-Y_{i}(0) .
$$
-  `the CATE function` is defined as

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

- If we denote our `learned estimates` as $\hat{\mu}_{0}(x)$ and $\hat{\mu}_{1}(x)$, then we can form the CATE estimate as the difference between the two
  $$
  \hat{\tau}(x)=\hat{\mu}_{1}(x)-\hat{\mu}_{0}(x) .
  $$

- In this work, we are interested in estimators with a small expected mean squared error (EMSE) for estimating the CATE,

$$
\operatorname{EMSE}(\mathcal{P}, \hat{\tau})=\mathbb{E}\left[(\tau(\mathcal{X})-\hat{\tau}(\mathcal{X}))^{2}\right] .
$$
> **HINT:** We can never observe the heterogeneous treatment effects because we can't observe the counterfactual, we can only estimate the heterogeneous treatment effects, so we can use the modified MSE formula:
> $$
> \operatorname{EMSE}(\mathcal{P}, \hat{\tau})=\mathbb{E}\left[(\tau(\mathcal{X})-\hat{\tau}(\mathcal{X}))^{2}-(\tau(\mathcal{X}))^{2}\right] .
> $$
> *(see Susan Atheya and Guido Imbensa,[《Recursive partitioning for heterogeneous causal effects》](https://www.pnas.org/doi/abs/10.1073/pnas.1510489113))*



## 2 Metaalgorithms

### 2.1 Ensemble Learning

> In [statistics](https://en.wikipedia.org/wiki/Statistics) and [machine learning](https://en.wikipedia.org/wiki/Machine_learning), **ensemble methods** use multiple learning algorithms to obtain better [predictive performance](https://en.wikipedia.org/wiki/Predictive_inference) than could be obtained from any of the constituent learning algorithms alone.



**Bagging (Bootstrap aggregating)**

Bagging使用装袋采样来获取数据子集训练基础学习器。通常分类任务使用投票的方式集成，而回归任务通过平均的方式集成。例如Random Forest、本文的metalearner等。

**Boosting**

Boosting指的是通过算法集合将弱学习器转换为强学习器。boosting的主要原则是训练一系列的弱学习器，所谓弱学习器是指仅比随机猜测好一点点的模型，例如较小的决策树，训练的方式是**利用加权的数据**。在训练的早期对于错分数据给予较大的权重。比较经典的有AdaBoost。



### 2.2 T-learner

<img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.37.40.png" alt="截屏2022-05-15 10.37.40" style="zoom:50%;" />

**Step1:** 用一个基学习器来拟合对照组的相应函数，![\mu_0(x)=\mathbb{E}[Y(0)|X=x]](https://math.jianshu.com/math?formula=%5Cmu_0(x)%3D%5Cmathbb%7BE%7D%5BY(0)%7CX%3Dx%5D)。基学习器可以在对照组的样本![{(X_i,Y_i)}_{W_i=0}](https://math.jianshu.com/math?formula=%7B(X_i%2CY_i)%7D_%7BW_i%3D0%7D)采用任何监督学习或者回归估计，我们用符号![\hat{\mu_0}](https://math.jianshu.com/math?formula=%5Chat%7B%5Cmu_0%7D)表示。

**Step2:** 我们估计treatment的相应函数，![\mu_1(x)=\mathbb{E}[Y(1)|X=x]](https://math.jianshu.com/math?formula=%5Cmu_1(x)%3D%5Cmathbb%7BE%7D%5BY(1)%7CX%3Dx%5D),在实验组的数据上进行训练，我们用符号![\hat{\mu_1}](https://math.jianshu.com/math?formula=%5Chat%7B%5Cmu_1%7D)来表示。T-learner可以通过以下公式得出：

![\hat{\tau_T}(x)=\hat{\mu_1}(x) -\hat{\mu_0}(x) \\ \tag{3}](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau_T%7D(x)%3D%5Chat%7B%5Cmu_1%7D(x)%20-%5Chat%7B%5Cmu_0%7D(x)%20%5C%5C%20%5Ctag%7B3%7D)



### 2.3 S-learner

<img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.38.05.png" alt="截屏2022-05-15 10.38.05" style="zoom:50%;" />

treatment是被当做特征使用的。评估公式为![\mu(x,w) := \mathbb{E}[Y^{abs}|X=x,W=w]](https://math.jianshu.com/math?formula=%5Cmu(x%2Cw)%20%3A%3D%20%5Cmathbb%7BE%7D%5BY%5E%7Babs%7D%7CX%3Dx%2CW%3Dw%5D),这里可以用任何基学习器，我们用![\hat{\mu}](https://math.jianshu.com/math?formula=%5Chat%7B%5Cmu%7D)表示模型的估计量，因此CATE估计量表示为：

![\hat{\tau_S}(x) =\hat{\mu}(x,1) -\hat{\mu}(x,0) \tag{4}](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau_S%7D(x)%20%3D%5Chat%7B%5Cmu%7D(x%2C1)%20-%5Chat%7B%5Cmu%7D(x%2C0)%20%5Ctag%7B4%7D)



### 2.4 X-learner

<img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.38.23.png" alt="截屏2022-05-15 10.38.23" style="zoom:50%;" />

**STEP 1:** 用任意的监督学习或者回归算法估计相应函数，用![\hat{\mu_0}](https://math.jianshu.com/math?formula=%5Chat%7B%5Cmu_0%7D)和![\hat{\mu_1}](https://math.jianshu.com/math?formula=%5Chat%7B%5Cmu_1%7D)表示估计量。

![\mu_0 = \mathbb{E}[Y(0)|X=x] \tag{5}](https://math.jianshu.com/math?formula=%5Cmu_0%20%3D%20%5Cmathbb%7BE%7D%5BY(0)%7CX%3Dx%5D%20%5Ctag%7B5%7D)![\mu_1 = \mathbb{E}[Y(1)|X=x] \tag{6}](https://math.jianshu.com/math?formula=%5Cmu_1%20%3D%20%5Cmathbb%7BE%7D%5BY(1)%7CX%3Dx%5D%20%5Ctag%7B6%7D)

**STEP 2:** 根据对照组的模型来计算实验组中的个人treatment效果，根据实验组的模型来计算实验最中的个人treatment效果。用公式表示为：

![\tilde{D_i^1} :=Y_i^1-\hat{\mu_0}(X_i^1) \tag{7}](https://math.jianshu.com/math?formula=%5Ctilde%7BD_i%5E1%7D%20%3A%3DY_i%5E1-%5Chat%7B%5Cmu_0%7D(X_i%5E1)%20%5Ctag%7B7%7D)
![\tilde{D_i^0} :=\hat{\mu_1}(X_i^0)-Y_i^0 \tag{8}](https://math.jianshu.com/math?formula=%5Ctilde%7BD_i%5E0%7D%20%3A%3D%5Chat%7B%5Cmu_1%7D(X_i%5E0)-Y_i%5E0%20%5Ctag%7B8%7D)

注意到，如果![\hat{\mu_0}=\mu_0](https://math.jianshu.com/math?formula=%5Chat%7B%5Cmu_0%7D%3D%5Cmu_0) 和![\hat{\mu_1}=\mu_1](https://math.jianshu.com/math?formula=%5Chat%7B%5Cmu_1%7D%3D%5Cmu_1)，则![\tau(x)=\mathbb{E}[\tilde{D^1}|X=x]=\mathbb{E}[\tilde{D^0}|X=x]](https://math.jianshu.com/math?formula=%5Ctau(x)%3D%5Cmathbb%7BE%7D%5B%5Ctilde%7BD%5E1%7D%7CX%3Dx%5D%3D%5Cmathbb%7BE%7D%5B%5Ctilde%7BD%5E0%7D%7CX%3Dx%5D)
  使用任意的监督学习或者回归算法计算![\tau(x)](https://math.jianshu.com/math?formula=%5Ctau(x))有两种方式：一种是利用treatment组训练的模型计算得到的![\hat{\tau_1}(x)](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau_1%7D(x)),另一种是利用对照组训练的模型计算得到的![\hat{\tau_0}(x)](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau_0%7D(x)).

**STEP 3:**  通过阶段2中计算得到的两个估计量进行加权计算CATE估计量：

<img src="/Users/zhugehuiming/Library/Application Support/typora-user-images/截屏2022-05-11 17.16.49.png" alt="截屏2022-05-11 17.16.49" style="zoom:60%;" />

![g \in [0,1]](https://math.jianshu.com/math?formula=g%20%5Cin%20%5B0%2C1%5D) 是一个权重函数。



### 2.5 Intuition Behind the Metalearners

- Fig. 1A shows the outcome for units in the treatment group (circles) and the outcome of units in the untreated group (crosses). **In this example, the CATE is constant and equal to one.**

<img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.38.55.png" alt="截屏2022-05-15 10.38.55" style="zoom:50%;" />

- The T-learner would now use estimator $\hat{\tau}_{T}(x)=\hat{\mu}_{1}(x)-\hat{\mu}_{0}(x)$ (Fig. 1C, solid line), ==which is a relatively complicated function== with jumps at 0 and 0.5, while the true  ${\tau}(x)$ is a constant. 
- For X-learner, If we choose $g(x)=\hat{e}(x)$, an estimator for the propensity score, ==$\hat{\tau}$ will be very similar to $\hat{\tau}_{1}(x)$==, since we have many more observations in the control group; i.e., $\hat{e}(x)$ is small.

`在这个例子中我们选择S-learner很难评估，例如当用RF的基学习器进行训练时，S-learner第一个split可能把97.5%的实验组的样本split出去，造成后续split时缺少实验组的样本。换句话说就是实验组和对照组的样本比例极不均衡时，如果使用S-learner训练时几次split就会把所有的实验组样本使用完。`



## 3 Simulation Results

**STEP 1:** First, we simulate a $d$-dimensional feature vector,
$$
X_{i} \stackrel{i i d}{\sim} \mathcal{N}(0, \Sigma)
$$
where $\Sigma$ is a correlation matrix that is created using the vine method (4).

**STEP 2:** Next, we create the potential outcomes according to
$$
\begin{aligned}
&Y_{i}(1)=\mu_{1}\left(X_{i}\right)+\varepsilon_{i}(1) \\
&Y_{i}(0)=\mu_{0}\left(X_{i}\right)+\varepsilon_{i}(0)
\end{aligned}
$$
where $\varepsilon_{i}(1), \varepsilon_{i}(0) \stackrel{i i d}{\sim} \mathcal{N}(0,1)$ and independent of $X_{i}$.

**STEP 3:** Finally, we simulate the treatment assignment according to
$$
W_{i} \sim \operatorname{Bern}\left(e\left(X_{i}\right)\right),
$$
we set $Y_{i}=Y\left(W_{i}\right)$, and we obtain $\left(X_{i}, W_{i}, Y_{i}\right) .^{\dagger}$
We train each CATE estimator on a training set of $N$ units, and we evaluate its performance against a test set of $10^{5}$ units for which we know the true CATE. We repeat each experiment 30 times, and we report the averages.

### 3.1 The unbalanced case with a simple CATE

We choose the propensity score to be constant and very small, e(x) = 0.01, such that on average only one percent of the units receive treatment.

**Simulation SI 1 (unbalanced treatment assignment).**
$$
\begin{aligned}
e(x) &=0.01, \quad d=20, \\
\mu_{0}(x) &=x^{T} \beta+5 \mathbb{I}\left(x_{1}>0.5\right), \quad \text { with } \beta \sim \operatorname{Unif}\left([-5,5]^{20}\right), \\
\mu_{1}(x) &=\mu_{0}(x)+8 \mathbb{I}\left(x_{2}>0.1\right) .
\end{aligned}
$$
<img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.40.00.png" alt="截屏2022-05-15 10.40.00" style="zoom:50%;" />

- The X-learner performs particularly well when the treatment group sizes are very unbalanced.



### 3.2 **Balanced cases without confounding**

Next, let us analyze two extreme cases: In one of them the CATE function is very complex and in the other one the CATE function is equal to zero.

1. Let us first consider the case where the treatment effect is as complex as the response functions in the sense that it does not satisfy regularity conditions (such as sparsity or linearity) that the response functions do not satisfy.

   **Simulation SI 2 (complex linear).**
   $$
   \begin{aligned}
   e(x) &=0.5, \quad d=20, \\
   \mu_{1}(x) &=x^{T} \beta_{1}, \text { with } \beta_{1} \sim \operatorname{Unif}\left([1,30]^{20}\right) \\
   \mu_{0}(x) &=x^{T} \beta_{0}, \text { with } \beta_{0} \sim \operatorname{Unif}\left([1,30]^{20}\right) .
   \end{aligned}
   $$
   The second setup (complex non-linear) is motivated by (3). Here the response function are non-linear functions.
   **Simulation SI 3 (complex non-linear).**
   $$
   \begin{aligned}
   e(x) &=0.5, \quad d=20 \\
   \mu_{1}(x) &=\frac{1}{2} \varsigma\left(x_{1}\right) \varsigma\left(x_{2}\right) \\
   \mu_{0}(x) &=-\frac{1}{2} \varsigma\left(x_{1}\right) \varsigma\left(x_{2}\right)
   \end{aligned}
   $$
   with
   $$
   \varsigma(x)=\frac{2}{1+e^{-12(x-1 / 2)}}
   $$
   <img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.40.51.png" alt="截屏2022-05-15 10.40.51" style="zoom:50%;" />

   - In this case, it is best to separate the CATE estimation problem into the two problems of estimating μ0 and μ1 since there is nothing one can learn from the other assignment group. ==The T-learner follows exactly this strategy and should perform very well==.

   - ==The S-learner, on the other hand, pools the data== and needs to learn that the response function for the treatment and the response function for the control group are very different.

   - Another interesting insight is that ==choosing BART or RF as the base learner can matter a great deal==. 

     

2. Let us now consider the other extreme where we choose the response functions to be equal. This leads to a zero treatment effect, which is very favorable for the S-learner.

   **Simulation SI 4 (global linear).**
   $$
   \begin{aligned}
   e(x) &=0.5, \quad d=5 \\
   \mu_{0}(x) &=x^{T} \beta, \text { with } \beta \sim \operatorname{Unif}\left([1,30]^{5}\right) \\
   \mu_{1}(x) &=\mu_{0}(x)
   \end{aligned}
   $$
   **Simulation SI 5 (piecewise linear).**
   $$
   \begin{aligned}
   e(x) &=0.5, \quad d=20, \\
   \mu_{0}(x) &= \begin{cases}x^{T} \beta_{l} & \text { if } x_{20}<-0.4 \\
   x^{T} \beta_{m} & \text { if }-0.4 \leq x_{20} \leq 0.4 \\
   x^{T} \beta_{u} & \text { if } 0.4<x_{20},\end{cases} \\
   \mu_{1}(x) &=\mu_{0}(x),
   \end{aligned}
   $$
   with
   $$
   \beta_{l}(i)=\left\{\begin{array}{ll}
   \beta(i) & \text { if } i \leq 5 \\
   0 & \text { otherwise }
   \end{array} \quad \beta_{m}(i)=\left\{\begin{array}{ll}
   \beta(i) & \text { if } 6 \leq i \leq 10 \\
   0 & \text { otherwise }
   \end{array} \quad \beta_{u}(i)= \begin{cases}\beta(i) & \text { if } 11 \leq i \leq 15 \\
   0 & \text { otherwise }\end{cases}\right.\right.
   $$
   and
   $$
   \beta \sim \operatorname{Unif}\left([-15,15]^{d}\right)
   $$
   <img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.41.57.png" alt="截屏2022-05-15 10.41.57" style="zoom:50%;" />
   
   - For both simulations, the CATE is globally 0. As expected, ==the S-learner performs very well==. Since the treatment indicator is given no special role, algorithms such as the lasso and RFs can completely ignore the treatment assignment by not choosing/splitting on it. This is beneficial if the CATE is in many places 0.



### 3.3 Confounding

In the preceding examples, the propensity score was globally equal to some constant. This is a special case, and in many observational studies, we cannot assume this to be true.

**Simulation SI 6 (beta confounded).**
$$
\begin{aligned}
e(x) &=\frac{1}{4}\left(1+\beta\left(x_{1}, 2,4\right)\right), \\
\mu_{0}(x) &=2 x_{1}-1 \\
\mu_{1}(x) &=\mu_{0}(x)
\end{aligned}
$$
<img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.42.29.png" alt="截屏2022-05-15 10.42.29" style="zoom:50%;" />

- Figure SI 5 shows that none of the algorithms performs significantly worse under confounding.



### 3.4 Comparison of Convergence Rates

In this section, we provide conditions under which the X-learner can be proven to outperform the T-learner in terms of pointwise estimation rate. (**details see[《Supporting Information: Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning》](https://www.pnas.org/doi/suppl/10.1073/pnas.1804597116)*)

**1. unbalanced groups**

In many real-world applications, we observe that the number of control units is much larger than the number of treated units, $m \gg n$. In that case, the bound on the EMSE of the T-learner will be dominated by the regression problem for the treated response function,
$$
\sup _{\mathcal{P} \in F} \operatorname{EMSE}\left(\mathcal{P}, \hat{\tau}_{T}^{n m}\right) \leq C_{1} n^{-a_{\mu}}
$$
The EMSE of the X-learner, however, will be dominated by the regression problem for the imputed treatment effects and it will achieve a faster rate of $n^{-a_{\tau}}$,
$$
\sup _{\mathcal{P} \in F} \operatorname{EMSE}\left(\mathcal{P}, \hat{\tau}_{X}^{n m}\right) \leq C_{2} n^{-a_{\tau}}
$$
This is a substantial improvement on when $a_{\tau}>a_{\mu}$, and it demonstrates that, in contrast to the $\mathrm{T}$-learner, the $\mathrm{X}$-learner can exploit structural conditions on the treatment effect. ==We therefore expect the X-learner to perform particularly well when one of the treatment groups is larger than the other==. 



**2. Example When the CATE Is Linear**

> (Lipschitz continuous regression functions). Let $F^{L}$ be the class of distributions on $(X, Y) \in[0,1]^{d} \times \mathbb{R}$ such that:
> 1. The features, $X_{i}$, are $i . i . d$. uniformly distributed in $[0,1]^{d}$.
> 2. The observed outcomes are given by
> $$
> Y_{i}=\mu\left(X_{i}\right)+\varepsilon_{i}
> $$
> where the $\varepsilon_{i}$ is independent and normally distributed with mean 0 and variance $\sigma^{2}$.
> 3. $X_{i}$ and $\varepsilon_{i}$ are independent.
> 4. The regression function $\mu$ is Lipschitz continuous with parameter L.

The optimal rate of convergence for the regression problem of estimating $x \mapsto \mathbb{E}[Y \mid X=x]$ in Definition SI 1 is $N^{-2 /(2+d)}$. Furthermore, the KNN algorithm with the right choice of the number of neighbors and the Nadaraya-Watson estimator with the right kernels achieve this rate, and they are thus minimax optimal for this regression problem.



**3. Other case**

If there are no regularity conditions on the CATE function and the response functions are Lipschitz continuous, then both the X- and T-learner obtain the same minimax optimal rate $\mathcal{O}\left(n^{2 /(2+d)}+m^{2 /(2+d)}\right)$.



## 4 Applications

### 4.1 Social Pressure and Voter Turnout

- In a large field experiment, Gerber et al. (1) show that **substantially higher turnout was observed among registered voters who received a mailing promising to publicize their turnout to their neighbors**.

- Fig. 2 presents the estimated treatment effects, using X-RF where the potential voters are grouped by their voting history. Fig. 2, Upper shows the proportion of voters with a significant positive (blue) and a significant negative (red) CATE estimate. ==We can see that there is evidence of a negative backlash among a small number of people who voted only once in thepast five elections before the general election in 2004==.
- Fig. 2, Lower shows the distribution of CATE estimates for each of the subgroups. ==If the number of mailers is limited, one should target potential voters who voted three times during the past five elections==, since this group has the highest ATE and it is a very big group of potential voters.

<img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.43.06.png" alt="截屏2022-05-15 10.43.06" style="zoom:50%;" />

- ==S-, T-, and X-RF all provide similar CATE estimates.== This is not surprising since the data set is very large and most of the covariates are discrete.

  <img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.43.31.png" alt="截屏2022-05-15 10.43.31" style="zoom:45%;" />



- We conducted a data-inspired simulation study to see how these estimators would behave in smaller samples. Fig. 3 presents the results of this simulation. They show that, ==in small samples, both X- and S-RF outperform T-RF, with X-RF performing the best==, as one may conjecture, given the unequal sample sizes.

  <img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.44.39.png" alt="截屏2022-05-15 10.44.39" style="zoom:50%;" />



**Supporting Information** 

(**details see[《Supporting Information: Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning》](https://www.pnas.org/doi/suppl/10.1073/pnas.1804597116)*)

<img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.45.35.png" alt="截屏2022-05-15 10.45.35" style="zoom:40%;" />

- none of the methods provide the correct coverage. The smooth CIs have a slightly higher coverage but the intervals are also slightly longer. However, the smooth CIs are computationally much more expensive and need a lot of bootstrap samples to be stable. ==Hence we prefer the normal approximated CIs==.
- We find that ==the coverage and the average confidence interval length for the overlap test set is very similar to that of the previous simulation study==, CI-Simulation 1. This is not surprising, because the two setups are very similar and the overlap condition is satisfied in both.
- We observe that for all methods the confidence intervals are tighter and the coverage is much lower than on the data where we have overlap because they try to extrapolate into regions of the covariate space without information on the treatment group. This is a problematic finding and suggests that ==confidence interval estimation in observational data is extremely difficult and that a violation of the overlap condition can lead to invalid inferences==.



### 4.2 Reducing Transphobia

- Broockman et al. (2, 27) show that **brief (10 min) but highquality door-to-door conversations can markedly reduce prejudice against gender-nonconforming individuals for at least 3 mo**.

- The authors find an ATE of 0.22 (SE: 0.072, t stat: 3.1) on their transgender tolerance scale. The authors report finding ==no evidence of heterogeneity in the treatment effect== that can be explained by the observed covariates. Their analysis is based on linear models (OLS, lasso, and elastic net) without basis expansions.
- Fig. 4A presents our results for estimating the CATE, using X–RF. We find that there is strong evidence that ==the positive effect that the authors find is only found among a subset of respondents that can be targeted based on observed covariates.== The average of our CATE estimates is within half a SD of the ATE that the authors report.

<img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.46.05.png" alt="截屏2022-05-15 10.46.05" style="zoom:50%;" />

- Fig. 4B presents the estimates from T–RF. These estimates are similar to those of X-RF, but with a larger spread.
- ==S-RF shrinks the treatment estimates toward zero.== The covariates are strongly predictive of the outcomes, and the splits in the S-RF are mostly on the features rather than the treatment indicator, because they are more predictive of the observed outcomes than the treatment assignment.

<img src="https://raw.githubusercontent.com/zhugehuiming2/ImageHost/main/img/%E6%88%AA%E5%B1%8F2022-05-15%2010.47.16.png" alt="截屏2022-05-15 10.47.16" style="zoom:40%;" />



## 5 Conclusion

- This paper reviewed metaalgorithms for CATE estimation including the S- and T-learners. It then introduced a metaalgorithm, the X-learner.

- Although none of the metaalgorithms is always the best, ==the X-learner performs well overall==, especially in the real-data examples.

- ==In practice, in finite samples, there will always be gains to be had if one accurately judges the underlying data-generating process.== For example, if the treatment effect is simple, or even zero, then pooling the data across treatment and control conditions will be beneficial when estimating the response model (i.e., the S-learner will perform well). However, if the treatment effect is strongly heterogeneous and the response surfaces of the outcomes under treatment and control are very different, pooling the data will lead to worse finite sample performance (i.e., the T-learner will perform well).



## Appendix: X-learner with R

`*This is an example from the tutorial of Prof. Susan Athey, Stanford University`

**Preparations**

**STEP 1: Fit the X-learner**

We will follow the algorithm outlined above very closely. Because the variable naming can be a bit cumbersome, let’s give some of our variables shorter aliases.

```R
X <- df_train[,covariate_names]
W <- df_train$W
Y <- df_train$Y
num.trees <- 200  #  We'll make this a small number for speed here.
n_train <- dim(df_train)[1]

# estimate separate response functions
tf0 <- regression_forest(X[W==0,], Y[W==0], num.trees=num.trees)
tf1 <- regression_forest(X[W==1,], Y[W==1], num.trees=num.trees)

# Compute the 'imputed treatment effects' using the other group
D1 <- Y[W==1] - predict(tf0, X[W==1,])$predictions
D0 <- predict(tf1, X[W==0,])$predictions - Y[W==0]

# Compute the cross estimators 
xf0 <- regression_forest(X[W==0,], D0, num.trees=num.trees)
xf1 <- regression_forest(X[W==1,], D1, num.trees=num.trees)

# Predict treatment effects, making sure to always use OOB predictions where appropriate
xf.preds.0 <- rep(0, n_train)
xf.preds.0[W==0] <- predict(xf0)$predictions
xf.preds.0[W==1] <- predict(xf0, X[W==1,])$predictions
xf.preds.1 <- rep(0, n_train)
xf.preds.1[W==0] <- predict(xf0)$predictions
xf.preds.1[W==1] <- predict(xf0, X[W==1,])$predictions

# Estimate the propensity score
propf <- regression_forest(X, W, num.trees=num.trees)
ehat <- predict(propf)$predictions

# Finally, compute the X-learner prediction
tauhat_xl <- (1 - ehat) * xf.preds.1 + ehat * xf.preds.0
```



**STEP 2: Predict point estimates**

The function `EstimateCate` provides point estimates. To predict on a test set:

```R
X.test <- df_test[,covariate_names]
ehat.test <- predict(propf, X.test)$predictions
xf.preds.1.test <- predict(xf1, X.test)$predictions
xf.preds.0.test <- predict(xf0, X.test)$predictions
tauhat_xl_test <- (1 - ehat.test) * xf.preds.1.test + ehat.test * xf.preds.0.test
```



**STEP 3: Compute confidence intervals**

Confidence intervals are computed via bootstrap. The process is straightforward but does not add any particular insight. We encourage the interested reader to see the algorithms in the paper for the exact implementation.



**(Preparations to be made)**

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
# R script for reading data from github repository, set path to where you have the tutorial files saved.
source('load_data.R') 

# Pick a dataset from the list above for parts I and II of the tutorial
df_experiment <- select_dataset("welfare")

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
                        
train_fraction <- 0.80  # Use train_fraction % of the dataset to train our models

df_train <- sample_frac(df, replace=F, size=train_fraction)
df_test <- anti_join(df,df_train, by = "ID")#need to check on larger datasets                  
```
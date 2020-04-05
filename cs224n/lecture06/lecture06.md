# Lecture06:Language Models and Recurrent Neural Networks

## Language Model

**Language Model：** is the task of predicting what word comes next.也可以认为语言模型是一个计算一段文本概率的系统。

### n-gram Language Model

Idea：Collect statistics about how frequent different n-grams are, and use these to predict next word.

假设：$x^{t+1}$只依赖于前n-1个词来预测。

所以由前n-1个词预测下一个词的概率为：

$$P(w|C)=\frac{count(C+w)}{count(C)}$$

稀疏问题（Sparsity Problems）
1、如果C+w没有出现过，那下一个词的概率就是0了。解决办法：对字典中每一个w加一个比较小的概率$\alpha$来**平滑**。
2、如果C没有出现过，就无法计算w的概率。解决办法：只用前n-2个词来预测，这称为**backoff**。

**n越大会使稀疏问题更为严重，一般n不超过5。**

### Neural Language Model


**优缺点：**
优点：
* 可以支持任意长度的文本
* 模型大小不随输入变化（参数共享）

缺点：
* 不可并行，计算速度慢。
* 很难利用长距离信息。

**反向传播**
BBPT：可以选择只向后传播n个状态，然后停止，否则可能需要传播很久。

**语言模型评价指标***

**perplexity**：混乱度。

$$perplexity=\prod_{t=1}^T(\frac{1}{P_{LM}(x^{t+1}|x^{t},...,x^1)})^{1/T}$$

$$=\prod_{t=1}^T(\frac{1}{\hat y_{t+1}^{t}})^{1/T}=exp(1/T\sum_{t=1}^T-\log y_{t+1}^t)=exp(J(\theta))$$

perplexity越小越好。

### 梯度消失与梯度爆炸

**梯度消失与梯度爆炸证明**
反向传播中的连乘效应。

**梯度消失**

* 值太小，反向传播会消失。
* 值太大，神经元饱和，梯度为0。
* 层级太多，梯度也会太小。

解决办法：

* 合适的激励函数
* 合适的参数初始化
* BN
* 残差连接网络 resnet
* 全连接Dense connections "DenseNet"

**梯度爆炸**

 clip

 ### LSTM

$$i_t=\sigma(W^{(i)}x_t+U^{(i)}h_{t-1})(Input gate)$$
$$f_t=\sigma(W^{(f)}x_t+U^{(f)}h_{t-1})(Forget gate)$$
$$o_t=\sigma(W^{(o)}x_t+U^{(o)}h_{t-1})(Output gate)$$
$$\tilde c_t=\tanh(W^{(c)}x_t+U^{(c)}h_{t-1})(New memory)$$
$$c_t=f_t\circ c_{t-1}+i_t\circ \tilde c_t(Final memory)$$
$$h_t=o_t\circ tanh(c_t)$$

 ### GRU

 $$z_t=\sigma(W^{(z)}x_t+U^{(z)}h_{t-1})(updata gate)$$
 $$r_t=\sigma(W^{(r)}x_t+U^{(r)}h_{t-1})(reset gate)$$
 $$\tilde h_t=tanh(r_t\circ Uh_{t-1}+Wx_t)(new memory)$$
 $$h_t=(1-z_t)\circ \tilde h_t+z_t\circ h_{t-1}$$
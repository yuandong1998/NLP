# 翻译模型

## 评价指标
### 1.PPL

Perplexity在自然语言领域衡量语言模型好坏的评价，翻译模型是语言模型的一种形式。它主要是根据每个词来估计一句话出现的概率，并用句子长度做normalize。

$$perplexity=\prod_{t=1}^T(\frac{1}{P_{LM}(x^{t+1}|x^{t},...,x^1)})^{1/T}$$

>可以理解为PPL越小，$p(w_i)$越大，一句我们期望的sentence出现的概率就越高。
>还有人说，Perplexity可以认为是average branch factor（平均分支系数），即预测下一个词时可以有多少种选择。别人在作报告时说模型的PPL下降到90，可以直观地理解为，在模型生成一句话时下一个词有90个合理选择，可选词数越少，我们大致认为模型越准确。这样也能解释，为什么PPL越小，模型越好。

perplexity另一种表达
$$=\prod_{t=1}^T(\frac{1}{\hat y_{t+1}^{t}})^{1/T}=exp(1/T\sum_{t=1}^T-\log y_{t+1}^t)=exp(J(\theta))$$

>这些是听报告了解的：
>1. 训练数据集越大，PPL会下降得更低，1billion dataset和10万dataset训练效果是很不一样的；
>2. 数据中的标点会对模型的PPL产生很大影响，一个句号能让PPL波动几十，标点的预测总是不稳定；
>3. 预测语句中的“的，了”等词也对PPL有很大影响，可能“我借你的书”比“我借你书”的指标值小几十，但从语义上分析有没有这些停用词并不能完全代表句子生成的好坏。

### 2.BLEU
#### 2.1 定义

**BLEU**:bilingual evaluation understudy，双语互译质量评估辅助工具。**BLUE去做判断：一句机器翻译的话与其相对应的几个参考翻译作比较，算出一个综合分数。这个分数越高说明机器翻译得越好。**

#### 2.2 优缺点
优点：方便、快速、结果有参考价值

缺点：
1. 不考虑语法上的准确性
2. 测评精度会受常用词的干扰
3. 短译句测评精度会较高
4. 没有考虑同义词或相似表达的情况，可能会导致合理翻译被否定

#### 2.3 实现

**公式**

$$BLEU=BP*exp(\sum_{n=1}^N w_n\log{P_n})$$

其中

$$BP=
\begin{cases}
1& \text{if c>r}\\
e^{1-r/c}& \text{if c<=r}
\end{cases}
$$

nltk.align.bleu_score模块实现了BLEU计算，包括三个函数：
```python
# 计算BLEU值
def bleu(candidate, references, weights)

# （1）私有函数，计算修正的n元精确率（Modified n-gram Precision）
def _modified_precision(candidate, references, n)

# （2）私有函数，计算BP惩罚因子
def _brevity_penalty(candidate, references)
```
**$P_n$的计算**

需要计算1-gram到4-gram。这里以1-gram为例：

候选译文（Predicted）：
It is a guide to action which ensures that the military always obeys the commands of the party

参考译文（Gold Standard）
1：It is a guide to action that ensures that the military will forever heed Party commands
2：It is the guiding principle which guarantees the military forces always being under the command of the Party
3：It is the practical guide for the army always to heed the directions of the party

首先统计候选译文里每个词出现的次数，然后统计每个词在参考译文中出现的次数，Max表示3个参考译文中的最大值，Min表示候选译文和Max两个的最小值。

![GD3eLn.png](https://s1.ax1x.com/2020/04/05/GD3eLn.png)

然后将每个词的Min值相加，将候选译文每个词出现的次数相加，然后两值相除即得

$P_1=\frac{3+0+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1}{3+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1}=0.95$
相应的可以计算出$P_2,P_3,P_4$，然后取w1=w2=w3=w4=0.25，也就是Uniform Weights。最后根据$\sum_{i=1}^N w_n\log{P_n}$计算结果。

**BP： Brevity Penalty 计算**

BP:Brevity Penalty(过短惩罚)，候选句子越短越接近与0。c为候选翻译句子长度，r为参考翻译中长度最接近候选翻译句子的长度。

最后将BP和P_n计算结合即可计算出BLEU。

**reference**
[1] [语言模型评价指标Perplexity——CSDN](https://blog.csdn.net/index20001/article/details/78884646)
[2] [BLEU原论文](https://www.aclweb.org/anthology/P02-1040.pdf)
[3] [机器翻译评价指标之BLEU详细计算过程](https://blog.csdn.net/guolindonggld/article/details/56966200)
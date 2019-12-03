# Attention机制
## 本质
视觉感知东西的时候注意特定部分。

## Attention机制的理解
Attention机制其实就是一系列注意力分配系数，也就是一系列权重参数罢了。

## Encoder-Decoder框架

![4857f6c3aa18407fd8ad5b9c5a9f6df1.png](en-resource://database/5586:1)
$$X=<X_1,X_2,...,X_m>$$
$$Y=<Y_1,Y_2,...,Y_n>$$

Encoder：
$$C=F(x_1,x_2,...,x_m)$$

Decoder：
$$y_i=g(C,y_1,y_2,...,y_{i-1})$$

这个模型，每个元素对语义编码的重要性程度都是一样的。所以要学习到Encoder重每个元素对Decoder重每个元素的重要性。

![3cf0dddbfd9874542fb92ca361c8f4ca.png](en-resource://database/5608:0)
$C_i$就变成了：
$$C_i = \sum_{j=0}^{Lx}a_{ij}f(x_j)$$


其中i表示时刻，j表示序列中第j个元素，$Lx$表示序列的长度，$f()$表示元素$x_j$的编码，$a_{ij}$就是指权重。
## attention函数
怎么得到attention value呢？

![1c6bdbd4629ff5b1036114cc0170e46e.jpeg](en-resource://database/5610:0)


attention value的本质：它其实就是一个查询(query)到一系列键值(key-value)对的映射。

![2f7bdbb98587c88ad6b1fd759d283b86.png](en-resource://database/5612:0)


attention函数共有三步完成得到attention value。
1. Q与K进行相似度计算得到权值
2. 对上部权值归一化
3. 用归一化的权值与V加权求和
此时加权求和的结果就为注意力值。

公式如下:
$$Attention Value = softmax(similarity(QK^T))V$$

在自然语言任务中，往往K和V是相同的。这时计算出的attention value是一个向量，代表序列元素$xj$的编码向量。此向量中包含了元素$xj$的上下文关系，即包含全局联系也拥有局部联系。


以seq2seq为例：

encoder：计算隐藏态 $(h_1,h_2,....,h_T)$
decoder：求$s_t$已知 $s_{t-1}$，每个输入位置j与当前输出关联性$e_{tj}=a(s_t-1,h_j)$，所以与每个位置关联性的向量为：

$$\vec e_t=(a(s_{t-1,h1}),...,a(s_{t-1},h_T))$$

$a$是相关性的算符
* 点乘 $\vec{e_t}=\vec s_{t-1}^T \vec h$
* 加权点乘 $\vec{e_t}=\vec s_{t-1}^T W \vech$
* 加和 $\vec e_t=\vec v^T \tanh(W_1\vec h+W_2\vec s_{t-1})$

再对$\vec e_t$进行softmax，$\vec\alpha_t=softmax(\vec e_t)$

最后计算出context vector  $\vec e_t=\sum_{j=1}^T \alpha_{tj}h_j$

计算decoder的下一个hidden state $$s_t=f(s_{t-1},y_{t-1,c_t})$$

得出 $p(y_|y_1,...,y_{t-1},\vec x) = g(y_{t-1},s_t,c_t)$

通过Attention权重矩阵的变化，更好地知道哪一部分翻译对应哪一部分源元素。

## 优劣
**优点**
* 一步到位的全局联系捕捉
   
   它先是进行序列的每一个元素与其他元素的对比，在这个过程中每一个元素间的距离都是一，因此它比时间序列RNNs的一步步递推得到长期依赖关系好的多，越长的序列RNNs捕捉长期依赖关系就越弱。
   
* 并行计算减少模型训练时间

    Attention机制每一步计算不依赖于上一步的计算结果，因此可以和CNN一样并行处理。但是CNN也只是每次捕捉局部信息，通过层叠来获取全局的联系增强视野。
    
* 模型复杂度小，参数少
    模型复杂度是与CNN和RNN同条件下相比较的。

**缺点**
    attention机制不是一个"distance-aware"的，它不能捕捉语序顺序(这里是语序哦，就是元素的顺序)。说到底，attention机制就是一个精致的"词袋"模型。所以有时候我就在想，在NLP任务中，我把分词向量化后计算一波TF-IDF是不是会和这种attention机制取得一样的效果呢? 当然这个缺点也好搞定，我在添加位置信息就好了。所以就有了 position-embedding(位置向量)的概念了





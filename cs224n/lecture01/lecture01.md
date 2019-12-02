#Lecture 01

## 词的表示

### WordNet

* missing nuance 无法表示细微的差别
* missing new meaning of words 
* Subjective
* 人力
* 无法计算相似性

### Onehot

* 维度高
* There is no natural notion of similarity for one-hot vectors! 任何两个词都是正交的，没有计算相似性的方法。

解决方法： learn to encode similarity in the vectors themselves 学习在向量本身中编码相似性

### Word vectors

word vectors\word embeddings\word representations : distributed representation

<u>Distributional semantics</u> : A word’s meaning is given
by the words that frequently appear close-by

**Word2vec**
* Go through each position t in the text, which has a center word
c and context (“outside”) words o

*  Use the similarity of the word vectors for c and o to calculate
the probability of o given c (or vice versa) **需要再理解** 根据c和o
的相似性计算给定c得到o的概率或者相反。

*  Keep adjusting the word vectors to maximize this probability




# Math Behind Attention

We implement the math behind the attention mechanism mentioned in *Attention Is All You Need* : https://arxiv.org/abs/1706.03762

There are two parts:

1. Attention with custom input embeddings (simple and small vectors). 
2. Attention with Word2vec embeddings.


![image.png](attachment:image.png)

We implement this part of the transformer's architecture from https://arxiv.org/pdf/1706.03762.pdf

In the first part, we simply define the input embeddings (vectors) manually. This is a simple approach just to carry out the math and write functions for reusability for future dense vectors (second part). The input embeddings are then introduced to positional encodings that preserve the position information of tokens in the input. The addition of positional encodings to input embeddings results to positional embeddings that act as input to the attention sub-layer. Inside the attention layer, each input is represented by Q, K, and V vectors (matrices) of model's dimension. Attention is calculated using the formula mentioned below that outputs a matrix containing values for each token with preserved model dimensions.

While in the second part, we write a simple sentence and generate its Word2Vec embeddings of dimension 224. These embeddings then go through positional encoding, eventually, leading to the generation of positional embeddings. Then, attention is calculated using multi-head attention mechanism. The positional embeddings of dimension 224 are divided into heads of 56 dimension. The output from all 4 heads, which is in dimension 56, is concatenated to form result of model dimension 224. This result is then normalised using the Post- Layer Norm method.

https://www.amazon.co.uk/Transformers-Natural-Language-Processing-architectures/dp/1803247339

## Libraries used

gensim==4.3.2
numpy==1.25.0
scipy==1.10.1


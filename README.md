# LoRA-Training-Different-LM-Modules





```
Input Text  
|  
Tokenization  
        |  
Input Indices  
|  
        v  
------------------------  
|   Embedding Layer    |  <------>  embedding input  
|   (embed_tokens)     |  
------------------------  
        |  
Input Vectors (Embeddings)  
|  
        v  
------------------------  
| Add & Normalize      |  <------>  normalization  
------------------------  
        |  
        v  
-----------------------  
| Multi-Head Attention |  <------>  multi-head self attention  
-----------------------  
|  
        v  
-----------------------------------  
|          Head 1 (并行)         |  
|---------------\                 |  
|              |                  |  
q_proj    k_proj    v_proj  
|              |                  |  
Query    Key      Value  
Vectors  Vectors  Vectors  
|              |                  |  
|<-------------|----------------->|  
|              |                  |  
|  Attention Scores (dot product) |  
|              |                  |  
|           Softmax               |  
|              |                  |  
|   Attention Distribution        |  
|              |                  |  
| Weighted Sum of Values          |  
|              |                  |  
|             o_proj              |  
-----------------------------------  
|  
        v  
-----------------------------------  
|          Head 2 (并行)         |  
|---------------\                 |  
|              |                  |  
q_proj    k_proj    v_proj  
|              |                  |  
Query    Key      Value  
Vectors  Vectors  Vectors  
|              |                  |  
|<-------------|----------------->|  
|              |                  |  
|  Attention Scores (dot product) |  
|              |                  |  
|           Softmax               |  
|              |                  |  
|   Attention Distribution        |  
|              |                  |  
| Weighted Sum of Values          |  
|              |                  |  
|             o_proj              |  
-----------------------------------  
|  
        v  
... (重复多个头，所有头并行计算)  
        |  
        v  
-----------------------------------  
| Concatenate Heads Output |  
-----------------------------------  
|  
        v  
-----------------------  
| Linear Projection    |  <------>  linear projection (o_proj)  
-----------------------  
        |  
Output Vectors  
|  
        v  
------------------------  
| Add & Normalize      |  <------>  add  
------------------------  
        |  
        v  
------------------------  
| Feed Forward Network |  <------>  feed forward  
------------------------  
|  
       fc1 (全连接层1)  
        |  
Activation (ReLU)  
|  
       fc2 (全连接层2)  
        |  
Output Vectors  
|  
        v  
------------------------  
| Add & Normalize      |  <------>  normalization  
------------------------  
        |  
        v  
---------------------------  
| Optional Gate Mechanism  |  
---------------------------  
|  
       gate_proj  
        |  
    Down Projection  
    (down_proj)  
|  
    Up Projection  
    (up_proj)  
        |  
Output Vectors  
|  
        v  
------------------------  
| Language Modeling Head |  <------>  language modeling head (lm_head)  
------------------------  
        |  
  Final Output
```

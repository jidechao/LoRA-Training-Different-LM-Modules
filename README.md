# LoRA-Training-Different-LM-Modules

In LoRA, besides the parameter target_modules, there is another parameter called modules_to_save. Through modules_to_save, we can specify which modules' parameters should be fully fine-tuned.

target_modules:

Purpose: Specify which modules to apply the LoRA technique for low-rank adaptation.
Behavior: Apply LoRA layers to these modules and update the parameters of these LoRA layers during the fine-tuning process.
modules_to_save:

Purpose: Specify which modules' weights need to be saved during the fine-tuning process.
Behavior: The weights of these modules will be saved throughout the fine-tuning process.
Next, we demonstrate full fine-tuning of Embeddings and the Language Modeling Head while fine-tuning the attention and MLP modules.

For example:

peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head","embed_tokens"],
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

Above code means full fine-tuning of the Embedding layers and the Language Modeling Head and LoRA training target_module togetherly.

Besides that, we could also LoRA training fc1 and fc2.

Let's take a look at the schematic diagram of the general relationships among these components.

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

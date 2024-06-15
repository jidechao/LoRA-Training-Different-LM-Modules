# LoRA-Training-Different-LM-Modules

In LoRA/QLoRA, there are two parameter target_modules and modules_to_save, with which we can specify which modules' parameters should be fully fine-tuned.

```
target_modules:

Purpose: Specify which modules to apply the LoRA technique for low-rank adaptation.
Behavior: Apply LoRA layers to these modules and update the parameters of these LoRA layers during the fine-tuning process.
```
```
modules_to_save:
Purpose: Specify which modules' weights need to be saved during the fine-tuning process.
Behavior: The weights of these modules will be saved throughout the fine-tuning process.
Next, we demonstrate full fine-tuning of Embeddings and the Language Modeling Head while fine-tuning the attention and MLP modules.
```
#### Note
It is important to note that when we want to add special tokens to the vocabulary, fully fine-tuning the token embeddings is crucial. If no special tokens are added, or if there is no need to fine-tune the existing tokens, then it might not be necessary to retrain the token embeddings. Generally, such operations are only performed when there is a significant domain knowledge transfer, such as adapting to languages other than English, or fine-tuning for rare text styles/types found on the internet.

For example:
```
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head","embed_tokens"],
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)
```
Above code means full fine-tuning of the Embedding layers and the Language Modeling Head and LoRA training target_module togetherly.

Besides that, we could also LoRA training fc1 and fc2.

## LM-Modules Relationship

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

## Training environment

Next, I will conduct LoRA training tests for five scenarios, using the model microsoft/Phi-3-medium-128k-instruct. The training code can be found in the code directory. The training environment utilizes Azure NC H100 GPU VMs. 

```
(headtraining) root@h100vm:~# nvidia-smi
Sat Jun 15 07:53:15 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 NVL                Off | 00000001:00:00.0 Off |                    0 |
| N/A   79C    P0             372W / 400W |  95247MiB / 95830MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      5240      C   ...iconda/envs/headtraining/bin/python    94482MiB |
+---------------------------------------------------------------------------------------+
```
## Training Results
Scenario 3 has the best training result.
Modules trained：

*modules_to_save=["lm_head","embed_tokens"],*

*target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]*

### Scenario 1 
Modules trained:

*modules_to_save=["lm_head"],*

*target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]*


The results of the training are as follows:


TrainOutput(global_step=154, training_loss=1.020355326788766, metrics={'train_runtime': 1991.3376, 'train_samples_per_second': 4.944, 'train_steps_per_second': 0.077, 'total_flos': 4.2289832773287936e+17, 'train_loss': 1.020355326788766, 'epoch': 1.0})

### Scenario 2 
Modules trained：

*modules_to_save=["embed_tokens"],*

*target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]*


The results of the training are as follows:






TrainOutput(global_step=154, training_loss=1.0223938551816074, metrics={'train_runtime': 1988.06, 'train_samples_per_second': 4.953, 'train_steps_per_second': 0.077, 'total_flos': 4.179327623626752e+17, 'train_loss': 1.0223938551816074, 'epoch': 1.0})


### Scenario 3 
Modules trained：

*modules_to_save=["lm_head","embed_tokens"],

*target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]*


The results of the training are as follows:


TrainOutput(global_step=154, training_loss=1.018851394777174, metrics={'train_runtime': 1995.55, 'train_samples_per_second': 4.934, 'train_steps_per_second': 0.077, 'total_flos': 4.2289832773287936e+17, 'train_loss': 1.018851394777174, 'epoch': 1.0})


### Scenario 4 
Modules trained：

*modules_to_save=["lm_head","embed_tokens"],*

*target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj","fc2","fc1"]*


The results of the training are as follows:


TrainOutput(global_step=154, training_loss=1.0191746284435321, metrics={'train_runtime': 1992.0904, 'train_samples_per_second': 4.943, 'train_steps_per_second': 0.077, 'total_flos': 4.2289832773287936e+17, 'train_loss': 1.0191746284435321, 'epoch': 1.0})



### Scenario 5
Modules trained：

*target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]*

The results of the training are as follows:


### Scenario 6
Modules trained：

*target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj","fc2","fc1"]*

The results of the training are as follows:


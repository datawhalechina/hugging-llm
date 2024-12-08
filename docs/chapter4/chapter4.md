

# 第4章 文本生成——文本任务都可生成

&emsp;&emsp;在上一章中，我们学习了如何使用大模型实现自然语言理解任务，包括文本分类、实体和关系抽取等，这些任务本质上是分类任务，也即将文本转化为结构化的表述。在理解文本的基础上，我们常常面临着更为复杂的任务，根据已有的文本去生成一段新的文本，这类任务被称作为NLG任务，它也是自然语言处理领域内的一个重要研究方向。    

&emsp;&emsp;事实上，绝大多数的自然语言处理任务都可以描述为自然语言生成任务，甚至是文本生成任务，也即将文本作为输入并将新的文本作为输出。举例来说，文本分类任务可以理解为输出类别名，如猫/狗、是/否；文本纠错任务可以理解为输入有错误的文本并理解，输出正确的文本描述；智能问答可以理解为根据背景知识及问句进行推理，输出相应的回答。

&emsp;&emsp;可以说，文本生成类任务的应用相当之广，本章将介绍一些常见的文本生成任务，主要包括文本摘要、文本纠错与机器翻译。其中包含曾经并不属于文本生成类任务，但如今也能使用NLG技术进行解决的任务，如文本纠错。

## 4.1 文本生成任务基础

&emsp;&emsp;我们可以认为文本分类任务的本质是，输入一段文本，并给定$N$类别选择，预测文本和每个类别的匹配概率，输出概率最高的类别。那么最简单的文本生成方式可以是，输入一段文本，并给定包含$N$个词的词表，在每个时刻根据当前已有文本，预测下一个词出现的概率，输出概率最高的词，这便是最早的语言模型。

```python
import numpy as np

# 定义词汇表和词频
vocab = ["我", "爱", "自然", "语言", "处理"]
word_freq = {"我": 0.1, "爱": 0.2, "自然": 0.3, "语言": 0.2, "处理": 0.2}
word_to_vec = {w: i for i, w in enumerate(vocab)}

next_word_prob = {
    "我": {"爱": 0.4, "自然": 0.3, "语言": 0.1, "处理": 0.2},
    "爱": {"我": 0.3, "自然": 0.3, "语言": 0.2, "处理": 0.2},
    "自然": {"我": 0.2, "爱": 0.2, "语言": 0.4, "处理": 0.2},
    "语言": {"我": 0.1, "爱": 0.1, "自然": 0.3, "处理": 0.5},
    "处理": {"我": 0.3, "爱": 0.2, "自然": 0.3, "语言": 0.2}
}

# 根据词频和词汇表选择下一个词
def select_next_word(current_word):
    next_word = np.random.choice(
        list(next_word_prob[current_word].keys()), 
        p=list(next_word_prob[current_word].values())
    )
    return next_word

# 生成文本序列并打印出来
text = w = "我"
for i in range(3):
    w = select_next_word(w)
    text += w

text == "我爱自然语言"
```

&emsp;&emsp;以上是一个简单的文本生成示例。我们首先给出包含$N$个词的词汇表，并给出给定一个词时出现下一个词的概率，这往往从语料库中的共现关系得到。在推理时，根据当前词汇和词频表，按照概率随机选择一个词作为输出。

&emsp;&emsp;当然，由于文本生成任务通常需要考虑上下文、语法结构等，单纯的基于概率的语言模型没法生成理想的文本，因此有了更多的基于深度学习的优化方法，例如编码器-解码器模型（encoder-decoder），BERT、GPT等预训练模型，生成对抗网络（generative adversarial networks）等。

&emsp;&emsp;在训练阶段，我们常常采用交叉熵损失来衡量生成的文本与真实文本之间的差异；在推理阶段，我们常常采用ROUGE（recall-oriented understudy for gisting evaluation）或BLEU（bilingual evaluation understudy）指标来评价生成文本的准确性与连贯性。关于评测部分，后续章节会有详细介绍。


## 4.2 文本摘要任务

### 4.2.1 什么是文本摘要？

&emsp;&emsp;文本摘要任务指的是用精炼的文本来概括整篇文章的大意，使得用户能够通过阅读摘要来大致了解文章的主要内容。

### 4.2.2 常见的文本摘要技术

&emsp;&emsp;从实现方法角度来看，文本摘要任务主要包括以下三种。

- 抽取式摘要：从原文档中提取现成的句子作为摘要句。
- 压缩式摘要：对原文档的冗余信息进行过滤，压缩文本作为摘要。
- 生成式摘要：基于NLG技术，根据源文档内容，由算法模型自己生成自然语言描述。

&emsp;&emsp;以下是一个基于mT5模型（T5模型的多语言版）的文本摘要样例。注意，模型较大，如果下载失败，可前往Huggingface官方网站搜索`mT5_multilingual_XLSum`模型，使用其提供的Hosted inference API进行测试。


```python
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
 
# 载入模型 
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

WHITESPACE_HANDLER = lambda k: re.sub("\s+", " ", re.sub("\n+", " ", k.strip()))

text = """自动信任协商主要解决跨安全域的信任建立问题,使陌生实体通过反复的、双向的访问控制策略和数字证书的相互披露而逐步建立信任关系。由于信任建立的方式独特和应用环境复杂,自动信任协商面临多方面的安全威胁,针对协商的攻击大多超出常规防范措施所保护的范围,因此有必要对自动信任协商中的攻击手段进行专门分析。按攻击特点对自动信任协商中存在的各种攻击方式进行分类,并介绍了相应的防御措施,总结了当前研究工作的不足,对未来的研究进行了展望"""
text = WHITESPACE_HANDLER(text)
input_ids = tokenizer(
    [text], return_tensors="pt", padding="max_length", truncation=True, max_length=512
)["input_ids"]

# 生成结果文本
output_ids = model.generate(input_ids=input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)[0]
output_text = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# 摘要文本
output_text == "自动信任协商 (AI) 是互信关系建立的最新研究工作的一部分。"
```

&emsp;&emsp;在上面的脚本中，我们首先从Huggingface官方网站下载`mT5_multilingual_XLSum`模型，这是mT5模型在多语言（multilingual）上的预训练模型，并基于XLSum文本摘要数据集进行了微调。对于一个输入文本，我们首先使用`tokenizer`将句子Token化并转化为对应的ID，再使用`model.generate`输出生成的Token ID列表，并使用`tokenizer`解码出对应的摘要文本。

&emsp;&emsp;可以看到，虽然我们使用了一个很复杂的模型，该模型也在摘要数据上进行了微调，单输出的结果仍旧不算十分完美。模型输出了更简短的文本，但是只总结了原文的第一句，对于后续提到的安全威胁、防御措施等，仅以“最新研究工作”一笔带过。

### 4.2.3 基于OpenAI接口的文本摘要实验

&emsp;&emsp;与前几章类似，我们调用OpenAI接口，利用大模型的内在理解能力，实现文本摘要功能。更进一步地，我们尝试使用OpenAI接口进行微调工作。

#### 1. 简单上手版：调用预训练模型

&emsp;&emsp;以下是调用基础版GPT模型实现文本摘要任务的样例，使用`openai.Completion.create`命令启动接口，并指定模型名称，将任务描述写入提示词中。值得注意的是，通过提示词控制字数并不一定准确。


```python
def summarize_text(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"请对以下文本进行总结，注意总结的凝炼性，将总结字数控制在20个字以内:\n{text}",
        temperature=0.3,
        max_tokens=500,
    )

    summarized_text = response.choices[0].text.strip()
    return summarized_text

text = "自动信任协商主要解决跨安全域的信任建立问题,使陌生实体通过反复的、双向的访问控制策略和数字证书的相互披露而逐步建立信任关系。由于信任建立的方式独特和应用环境复杂,自动信任协商面临多方面的安全威胁,针对协商的攻击大多超出常规防范措施所保护的范围,因此有必要对自动信任协商中的攻击手段进行专门分析。按攻击特点对自动信任协商中存在的各种攻击方式进行分类,并介绍了相应的防御措施,总结了当前研究工作的不足,对未来的研究进行了展望。"""
output_text = summarize_text(text)
# 摘要文本
output_text == "自动信任协商解决跨安全域信任建立问题，但面临多种安全威胁，需要分析攻击方式及防御措施。"
# 摘要文本长度
len(output_text) == 43
```

&emsp;&emsp;接下来，我们尝试调用ChatGPT实现相同的功能。


```python
def summarize_text(text):
    content = f"请对以下文本进行总结，注意总结的凝炼性，将总结字数控制在20个字以内:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": content}],
        temperature=0.3
    )
    summarized_text = response.get("choices")[0].get("message").get("content")
    return summarized_text

text = """自动信任协商主要解决跨安全域的信任建立问题,使陌生实体通过反复的、双向的访问控制策略和数字证书的相互披露而逐步建立信任关系。由于信任建立的方式独特和应用环境复杂,自动信任协商面临多方面的安全威胁,针对协商的攻击大多超出常规防范措施所保护的范围,因此有必要对自动信任协商中的攻击手段进行专门分析。按攻击特点对自动信任协商中存在的各种攻击方式进行分类,并介绍了相应的防御措施,总结了当前研究工作的不足,对未来的研究进行了展望。"""
output_text = summarize_text(text)
# 摘要文本
output_text == "自动信任协商解决跨域信任建立，但面临多方面安全威胁，需分类防御。研究不足，未来展望。"
# 摘要文本长度
len(output_text) == 42
```

&emsp;&emsp;总体来说，两个接口在未经过微调的文本摘要任务上，已经表现出了比mT5更为优秀的效果。对于生成任务，每次输入相同的问题，输出的结果都可能存在一定的随机性，我们也可以称之为创造性，可由`temperature`参数控制创造性程度，参数越高则模型输出的自由度越高。对于文本摘要、纠错、翻译类任务，我们希望的输出偏向于标准的答案，因此`temperature`可以设置的更低一些；而对于续写小说之类的任务，我们希望的输出可能是天马行空的，因此`temperature`可以设置的更高一些。

#### 2. 进阶优化版：基于自定义语料微调

&emsp;&emsp;对于垂直领域的数据或任务，有时直接使用大语言模型的效果不佳。当然，由于ChatGPT强大的内在理解能力，在某些情况下使用一个比较好的提示词，通过零样本或者少样本也能得到一个不错的结果。我们使用CSL摘要数据集，基于ada模型为例，简单介绍如何通过自定义语料库对模型进行微调。

&emsp;&emsp;CSL摘要数据集是计算机领域的论文摘要和标题数据，包含3500条数据，其中标题数据的平均字数为18，字数标准差为4，最大字数为41，最小数字为6；论文摘要数据的平均字数为200，字数标准差为63，最大字数为631，最小数字为41。

```python
import json
with open("dataset/csl_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
```

&emsp;&emsp;先读取数据集，其中一条样例数据如下所示。


```python
data[-1] == {
    "title": "自动信任协商中的攻击与防范",
    "content": "自动信任协商主要解决跨安全域的信任建立问题,使陌生实体通过反复的、双向的访问控制策略和数字证书的相互披露而逐步建立信任关系。由于信任建立的方式独特和应用环境复杂,自动信任协商面临多方面的安全威胁,针对协商的攻击大多超出常规防范措施所保护的范围,因此有必要对自动信任协商中的攻击手段进行专门分析。按攻击特点对自动信任协商中存在的各种攻击方式进行分类,并介绍了相应的防御措施,总结了当前研究工作的不足,对未来的研究进行了展望。"
}
```

&emsp;&emsp;接着，我们需要将自定义语料数据集转化成OpenAI所需要的标准格式。OpenAI提供了一个数据准备工具`fine_tunes.prepare_data`，我们只需要将数据集整理成它要求的格式，第一列列名为`prompt`，表示输入文本，第二列列名为`completion`，表示输出文本，然后将其保存为`json`格式，一行为一个记录，即可使用该数据准备工具。


```python
import pandas as pd

df = pd.DataFrame(data)
df = df[["content", "title"]]
df.columns = ["prompt", "completion"]
df_train = df.iloc[:500]
df_train.head(5)
```

&emsp;&emsp;构造好的训练数据样例如表4-1所示。


<center>表4-1 CSL摘要数据集样例</center>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>completion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>提出了一种新的保细节的变形算法,可以使网格模型进行尽量刚性的变形,以减少变形中几何细节的扭曲...</td>
      <td>保细节的网格刚性变形算法</td>
    </tr>
    <tr>
      <th>1</th>
      <td>实时服装动画生成技术能够为三维虚拟角色实时地生成逼真的服装动态效果,在游戏娱乐、虚拟服装设计...</td>
      <td>一种基于混合模型的实时虚拟人服装动画方法</td>
    </tr>
    <tr>
      <th>2</th>
      <td>提出一种基于模糊主分量分析技术(FPCA)的人脸遮挡检测与去除方法.首先,有遮挡人脸被投影到...</td>
      <td>人脸遮挡区域检测与重建</td>
    </tr>
    <tr>
      <th>3</th>
      <td>图像匹配技术在计算机视觉、遥感和医学图像分析等领域有着广泛的应用背景.针对传统的相关匹配算法...</td>
      <td>一种基于奇异值分解的图像匹配算法</td>
    </tr>
    <tr>
      <th>4</th>
      <td>提出了一种基于片相似性的各项异性扩散图像去噪方法.传统的各项异性图像去噪方法都是基于单个像素...</td>
      <td>片相似性各项异性扩散图像去噪</td>
    </tr>
  </tbody>
</table>

&emsp;&emsp;将`DataFrame`保存成`jsonl`格式，注意，由于数据集中存在中文，使用常规的`ascii`编码可能会出现编译问题，可以设置参数`force_ascii=False`，如下所示。


```python
df_train.to_json("dataset/csl_summarize_finetune.jsonl", orient="records", lines=True, force_ascii=False)
```

&emsp;&emsp;调用`fine_tunes.prepare_data`工具，在处理数据的过程中，该工具会自动根据数据情况做一些转换，例如将输入输出转化为小写，在`prompt`后增加`->`符号，在`completions`后加`\n`标识等。这些在“句词分类”一章中有提到，读者可以结合两章内容回顾知识。


```python
!openai tools fine_tunes.prepare_data -f dataset/csl_summarize_finetune.jsonl -q
```

&emsp;&emsp;输出日志样例如下所示。

```python
Analyzing...
(...)  # 省略打印
Based on the analysis we will perform the following actions:
- [Recommended] Lowercase all your data in column/key `prompt` [Y/n]: Y
- [Recommended] Lowercase all your data in column/key `completion` [Y/n]: Y
- [Recommended] Add a suffix separator ` ->` to all prompts [Y/n]: Y
- [Recommended] Add a suffix ending `\n` to all completions [Y/n]: Y
- [Recommended] Add a whitespace character to the beginning of the completion [Y/n]: Y
Your data will be written to a new JSONL file. Proceed [Y/n]: Y
Wrote modified file to `dataset/csl_summarize_finetune_prepared.jsonl`
(...)
```

 &emsp;&emsp;当上述脚本执行完后，在`dataset/`文件夹下，我们会发现一个新产生的文件`csl_summarize_finetune_prepared.jsonl`，这便是处理好的标准化的数据文件。接着我们创建一个微调任务，指定数据集和模型，OpenAI会自动上传数据集并开始微调任务。


```python
import openai
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

!openai api fine_tunes.create \
    -t "./dataset/csl_summarize_finetune_prepared.jsonl" \
    -m ada\
    --no_check_if_files_exist
```

&emsp;&emsp;执行命令后，输出的日志如下所示。

    Uploaded file from ./dataset/csl_summarize_finetune_prepared.jsonl: file-gPzuOBUizUDCGO7t0oDYoWQB
    
    Upload progress:   0%|          | 0.00/380k [00:00<?, ?it/s]
    Upload progress: 100%|██████████| 380k/380k [00:00<00:00, 239Mit/s]
    
    Created fine-tune: ft-LoKi6mOxlkOtfZcZTrmivKDa
    Streaming events until fine-tuning is complete...    
    (Ctrl-C will interrupt the stream, but not cancel the fine-tune)
    [2023-05-07 20:27:26] Created fine-tune: ft-LoKi6mOxlkOtfZcZTrmivKDa
    [2023-05-07 20:27:45] Fine-tune costs $0.43
    [2023-05-07 20:27:45] Fine-tune enqueued. Queue number: 0
    [2023-05-07 20:27:46] Fine-tune started
    (...)

&emsp;&emsp;根据上一步的输出，得到微调运行的key：`ft-LoKi6mOxlkOtfZcZTrmivKDa`，同时日志中也会给出预估的微调任务成本，比如这里是0.43美元。我们可以通过`get`命令来获取当前执行进度。当从日志中找到`fine_tuned_model`，且`status`为`succeeded`时，则表明任务已经执行成功。


```python
!openai api fine_tunes.get -i ft-LoKi6mOxlkOtfZcZTrmivKDa
```

&emsp;&emsp;成功日志如下所示。

    {
    (...)
      "fine_tuned_model": "ada:ft-personal-2023-04-15-13-29-50",
    (...)
      "status": "succeeded",
    }

&emsp;&emsp;还可以通过`fine_tunes.results`来保存训练过程的记录，从而帮助我们更好地监控模型的运行情况。

```python
# 保存openai fine tune过程的记录
!openai api fine_tunes.results -i ft-LoKi6mOxlkOtfZcZTrmivKDa > dataset/metric.csv
```

&emsp;&emsp;在微调完成后，就可以像使用ChatGPT一样方便地使用自己的微调模型，只需要将模型名称修改为刚才微调好的模型即可，如下所示。


```python
def summarize_text(text, model_name):
    response = openai.Completion.create(
        engine=model_name,
        prompt=f"请对以下文本进行总结，注意总结的凝炼性，将总结字数控制在20个字以内:\n{text}",
        temperature=0.3,
        max_tokens=100,
    )

    summarized_text = response.choices[0].text.strip()
    return summarized_text

text = "自动信任协商主要解决跨安全域的信任建立问题,使陌生实体通过反复的、双向的访问控制策略和数字证书的相互披露而逐步建立信任关系。由于信任建立的方式独特和应用环境复杂,自动信任协商面临多方面的安全威胁,针对协商的攻击大多超出常规防范措施所保护的范围,因此有必要对自动信任协商中的攻击手段进行专门分析。按攻击特点对自动信任协商中存在的各种攻击方式进行分类,并介绍了相应的防御措施,总结了当前研究工作的不足,对未来的研究进行了展望。"""

ada_abs = summarize_text(text, model_name="ada")
ada_ft_abs = summarize_text(text, model_name="ada:ft-personal-2023-04-15-13-29-50")
# ada摘要文本
ada_abs == "因此,为了在未来进行研究,本次研究也许能给学术界其他学者带来建议,更多读者本次研究期间的能查"
# ada微调模型摘要文本
ada_ft_abs == """分布式防御措施的自动信任协商

面向自动信任协商的防御措施研究

自动信任协商的攻击面临"""
```

&emsp;&emsp;由于资费与效率原因，本次实验基于`ada`模型进行微调。可以看到，原始的`ada`模型几乎完全没有理解文本摘要任务的需求，只是在文本背景上生成了一段新的文本。在经过简单的微调后，相比原始模型已经有了质的飞跃，并且在一定程度上能生成一个可用的摘要。不过由于我们只使用了500条样本进行微调实验，模型的微调效果有限，生成的文本仍然远不及ChatGPT或者其他在该任务上做过精细微调的大模型，如需进一步优化，可以增加训练样本的数量与质量，或者换一个更好的基础模型，这也会带来一定的训练成本增加。

&emsp;&emsp;如果需要在一个微调模型上继续微调，直接将`fine_tunes.create`的`-m`参数改为微调后的模型名称即可，如下所示。


```python
!openai api fine_tunes.create \
    -t "./dataset/csl_summarize_finetune_prepared.jsonl" \
    -m ada:ft-personal-2023-04-15-13-29-50\
    --no_check_if_files_exist
```

&emsp;&emsp;我们可以通过`fine_tunes.list`查看所有微调模型，也可以通过`openai.Model.list()`查看名下所有可支持的模型，这里面会包含所有训练成功的微调模型。

```python
# 查看所有的fine tune模型
!openai api fine_tunes.list
```

&emsp;&emsp;这条命令会输出一个模型信息列表，列表的每个元素是类似下面示例的一个字典，包含了创建时间、模型名称、模型超参数、模型ID、基础模型名称、训练文件、执行状态等。每一个我们训练的模型，不管是成功还是失败，均会在这里展示出来。

```
{
  "created_at": 1681565036,
  "fine_tuned_model": "ada:ft-personal-2023-04-15-13-29-50",
  "hyperparams": {
    "batch_size": 1,
    "learning_rate_multiplier": 0.1,
    "n_epochs": 4,
    "prompt_loss_weight": 0.01
  },
  "id": "ft-LoKi6mOxlkOtfZcZTrmivKDa",
  "model": "ada",
  "object": "fine-tune",
  (...)
}
```

&emsp;&emsp;可以查看可用的模型，其中包含自己微调的模型，以`ft-personal`开头。


```python
models = openai.Model.list()
[x.id for x in models.data] == [
    "babbage", 
    "davinci", 
    ...,
    "ada:ft-personal-2023-05-07-07-50-50", 
    "ada:ft-personal-2023-04-15-13-19-25", 
    "ada:ft-personal-2023-04-15-13-29-50"
]
```

&emsp;&emsp;如需删除自己微调的模型，可以使用`openai.Model.delete`命令。


```python
openai.Model.delete("ada:ft-personal-2023-04-15-12-54-03")
```

&emsp;&emsp;OpenAI的官方指引提供了很多微调相关的参数与指令说明，感兴趣的读者可以在官网获取更详细的指导。


## 4.3 文本纠错任务

### 4.3.1 什么是文本纠错？

&emsp;&emsp;在日常生活中，不管是微信聊天、微博推文甚至是出版书籍中，我们都或多或少地发现文本中的错别字现象。这些错别字可能源于语音输入时的口音偏差，如“飞机”被输入成了“灰机”；也可能是拼音输入时误触了临近键位或者选错了结果，如“飞机”被输入成了“得急”、“肥鸡”；亦或是手写输入时写成了形近字，如“战栗”被写成了“战粟”。常见的错误类型包括下面几种。

- 拼写错误：如中文课程->中文磕碜。明天会议->明天会易。
- 语法错误：他昨天去参加会议了。->他昨天将要去参加会议。
- 标点符号错误：您好，请多指教！->您好，请多指教???
- 知识性错误：上海黄浦区->上海黄埔区。
- 重复性错误：您好，请问您今天有空吗？->您好，请问您今天有空吗吗吗吗吗吗？
- 遗漏性错误：他昨天去参加会议了。->他昨天去参加了。
- 语序性错误：他昨天去参加会议了。->他昨天去会议参加了。
- 多语言错误：他昨天去参加会议了。->他昨天去参加huiyi了。

&emsp;&emsp;总之，文本错误可能是千奇百怪的。对于人类而言，凭借常识与上下文，实现语义理解尚不是什么难事，有时只是些许影响阅读体验。但对于一些特定的文本下游任务，如命名实体识别或意图识别，一条不加处理的错误输入文本可能会导致南辕北辙的识别结果。

&emsp;&emsp;文本纠错任务指的是通过自然语言处理技术对文本中出现的错误进行检测和纠正的过程。目前已经成为自然语言处理领域中的一个重要分支，被广泛地应用于搜索引擎、机器翻译、智能客服等各种场景。纵然由于文本错误的多样性，我们往往难以将所有错误通通识别并纠正成功，但是如果能尽可能多且正确地识别文本中的错误，能够大大降低人工审核的成本，也不失为一桩美事。

### 4.3.2 常见的文本纠错技术

&emsp;&emsp;常见的文本纠错技术主要有以下几种。

- 基于规则的文本纠错技术。

- 基于语言模型的文本纠错技术。

- 基于掩码语言模型（mask language model，MLM）的文本纠错技术。

- 基于NLG的文本纠错技术。

&emsp;&emsp;下面，我们对这几种技术进行详细的阐述。

#### 1. 基于规则的文本纠错技术

&emsp;&emsp;这种文本纠错技术是通过实现定义的规则来检查文本中的拼写、语法、标点符号等常见错误，比如“金字塔”常被误写为“金子塔”，则在数据库中加入两者的映射关系。由于这种传统方法需要大量的人工工作以及专家对于语言的深刻理解，因此难以处理海量文本或较为复杂的语言错误。

#### 2. 基于语言模型的文本纠错技术

&emsp;&emsp;基于语言模型的文本纠错技术包括错误检测和错误纠正，这种方法同样比较简单粗暴，方法速度快，扩展性强，但效果一般。常见的模型有Kenlm。

- 错误检测：使用类似结巴分词等分词工具对句子进行切词，然后结合字粒度和词粒度两方面得到疑似错误结果，形成疑似错误位置候选集。

- 错误纠正：遍历所有的候选集并使用音似、形似词典替换错误位置的词，然后通过语言模型计算句子困惑度（一般来说，句子越通顺，困惑度越低），最后比较并排序所有候选集结果，得到最优纠正词。

#### 3. 基于掩码语言模型的文本纠错技术

&emsp;&emsp;BERT在预训练阶段使用了掩码语言模型和下一句预测（next sentence prediction，NSP）两个任务。其中掩码语言模任务类似于英文的完形填空，在一段文本中随机遮住一个词，让模型通过上下文语境来预测这个词是什么；下一句预测任务则是给定两个句子，判断一个句子是否是另一个句子的下一句，从而帮助模型理解上下文的语义连贯性。在BERT的后续改进模型中，RobertA中将下一句预测任务直接放弃，ALBERT则将下一句预测替换成句子顺序预测（sentence order prediction，SOP）。这些操作表明，下一句预测任务作为一个分类任务，是相对简单的，BERT的主要能力来源于掩码语言模型。

&emsp;&emsp;在掩码语言模型任务的训练阶段，有15%的词会被遮掩，这其中80%的词汇被替换为`[MASK]`特殊符号标识，10%被替换成随机的其他词汇，10%仍旧保持不变。从而，总共有15%×10%的词汇会被替换为随机的其他词汇，迫使模型更多地依赖于上下文信息去预测遮掩词汇，在一定程度上赋予了模型纠错能力。

&emsp;&emsp;因此，我们将BERT的掩码语言模型任务做一下简单的修改，将输入设计为错误的词汇，输出为正确的词汇，做一下简单的微调，即可轻松实现文本纠错功能。比如Soft-Masked BERT模型，设计了一个二重网络来进行文本纠错，其中“错误检测网络”通过一个简单的双向语言模型判断每个字符错误的概率，“错误纠正网络”将错误概率更高的词进行遮掩，并预测出真实词汇。

&emsp;&emsp;以下是一个基于Huggingface的`MacBERT4CSC`进行纠错的样例。注意，`MacBERT4CSC`会自动将所有的英文字符转为小写，并且我们查看修改时会忽略大小写上的差异。

```python
from transformers import BertTokenizer, BertForMaskedLM

# 载入模型
tokenizer = BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")

text = "大家好,一起来参加DataWhale的《ChatGPT使用指南》组队学习课乘吧！"
input_ids = tokenizer([text], padding=True, return_tensors="pt")

# 生成结果文本
with torch.no_grad():
    outputs = model(**input_ids)
output_ids = torch.argmax(outputs.logits, dim=-1)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(" ", "")
# 纠错文本
output_text == "大家好,一起来参加datawhale的《chatgpt使用指南》组队学习课程吧！"
```

&emsp;&emsp;进一步地，我们可以通过以下脚本来展示修改的位置。

```python
# 查看修改
import operator

def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if ori_char in [" ", "“", "”", "‘", "’", "琊", "\n", "…", "—", "擤"]:
            # add unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if i >= len(corrected_text):
            continue
        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details

correct_text, details = get_errors(output_text[:len(text)], text)
details == [("乘", "程", 37, 38)]
```

#### 4. 基于NLG的文本纠错技术

&emsp;&emsp;上述提到的掩码方法只能用于输入与输出等长的情况，但是实际应用中往往会出现两者不等长的情况，如错字或多字。一种可能的解决办法是，在原有的BERT模型后嵌入一层Transformer解码器，即将“文本纠错”任务等价成“将错误的文本翻译成正确的文本”。不过此时我们没法保证输出文本与原始文本中正确的部分一定能保持完全一致，可能会在语义不变的情况下，生成一种新的表达方式。

### 4.3.3 基于OpenAI接口的文本纠错实验

&emsp;&emsp;我们直接尝试使用ChatGPT来进行文本纠错，如下所示。


```python
def correct_text(text):
    content = f"请对以下文本进行文本纠错:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": content}]
    )
    corrected_text = response.get("choices")[0].get("message").get("content")
    return corrected_text

text = "大家好,一起来参加DataWhale的《ChatGPT使用指南》组队学习课乘吧！"
output_text = correct_text(text)
# 纠错文本
output_text == "大家好，一起来参加DataWhale的《ChatGPT使用指南》组队学习课程吧！"
```

&emsp;&emsp;类似于上文的查看修改位置脚本，我们可以使用`Redlines`函数来实现类似的功能。具体来说，就是对比输入文本和输出文本之间的差异，用划线与标红来表示差异点。

```python
from redlines import Redlines
from IPython.display import display, Markdown

diff = Redlines(" ".join(list(text)), " ".join(list(output_text)))
display(Markdown(diff.output_markdown))
```

&emsp;&emsp;结果如下所示（注意，这里添加了空格），可以发现连标点都给修正过来了。

&emsp;&emsp;大 家 好 <span style="color:grey;text-decoration:line-through;">, </span><span style="color:grey;font-weight:700;">， </span>一 起 来 参 加 D a t a W h a l e 的 《 C h a t G P T 使 用 指 南 》 组 队 学 习 课 <span style="color:grey;text-decoration:line-through;">乘 </span><span style="color:grey;font-weight:700;"><i>程 </i></span>吧 ！


## 4.4 机器翻译任务

### 4.4.1 什么是机器翻译？

&emsp;&emsp;机器翻译，又称为自动翻译，是利用计算机将一种自然语言（源语言）转换为另一种自然语言（目标语言）的过程。据不完全统计，世界上约有7000种语言，两两配对约有4900万种组合，这些语言中又不乏一词多义、垂类知识等现象。因此，能够使用更少的标注数据，或者无监督地让计算机真正地理解输入语言的含义，并“信”、“达”、“雅”地转化为输出语言，是历来学者们的研究重心。

&emsp;&emsp;众所周知，机器翻译一直是自然语言处理领域备受关注的研究方向，也是自然语言处理技术最早展露头角的任务之一。如今市面上的机器翻译工具层出不穷，如大家常用的百度翻译、谷歌翻译，乃至小时候科幻片里才有的AI同声传译，如讯飞听见同传。简单来说可以将其划分为通用领域（多语种）、垂直领域、术语定制化、领域自适应、人工适应、语音翻译等。

### 4.4.2 常见的机器翻译技术

&emsp;&emsp;从机器翻译的发展历程来看，主要经历了如下几个阶段。

- 基于规则的方法。
- 基于统计的方法。
- 基于神经网络的方法。

#### 1. 基于规则的机器翻译技术

&emsp;&emsp;基于规则的方法需要建立各类知识库，描述源语言和目标语言的词法、句法以及语义知识。简单来说就是建立一个翻译字典与一套语法规则，先翻译重要的词汇，再根据目标语言的语法将词汇拼接成正确的句子。这种方法需要丰富且完善的专家知识，且对于未在字典及规则中出现过的情况，则无法处理。

#### 2. 基于统计的机器翻译技术

&emsp;&emsp;基于统计的方法则是从概率的角度去实现翻译，其核心原理是，对于源语言R中的每个词$r$，从此表中找到最可能与之互译的单词$t$，再调整单词$t$的顺序，使其合乎目标语言T的语法。假设我们拥有一个双语平行语料库，可以通过源词与目标词在两个句子中共同出现的频率作为两个词表示的是同一个词的概率。比如将“我对你感到满意”翻译成英文，假设中文的“我”和英文的“I”、“me”、“I'm”共同出现的概率最高，也即它们表示的是同一个词的概率最高，我们将其作为候选词，再根据英文语法挑选出“I'm”是最佳的翻译词。这被称之为基于词对齐的翻译方法。但是由于短语和语法的存在，有时并不是一个单词表示一个含义，而是一个短语共同组合表示一个含义，如英文的“a lot of”共同表示了中文的“很多”。因此，将翻译的最小单位设计成词显然是不符合语法的，后来又延申出了基于短语的翻译方法，将最小翻译单位设计成连续的词串。

#### 3. 基于神经网络的机器翻译技术

&emsp;&emsp;2013年，一种用于机器翻译的新型端到端编码器-解码器架构问世，将CNN用于隐含表征挖掘，将RNN用于将隐含向量转化为目标语言，标志了神经机器翻译开端。后来，Attention、Transformer、BERT等技术被相继提出，大大提升了翻译的质量。

&emsp;&emsp;以下是一个基于`transformers`实现机器翻译的简单示例。


```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

text = "大家好，一起来参加DataWhale的《ChatGPT使用指南》组队学习课程吧！"

inputs = tokenizer(text, return_tensors="pt", )
outputs = model.generate(inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True)
translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
# 翻译文本
translated_sentence == "Hey, guys, let's join the ChatGPT team at DataWhale."
```

&emsp;&emsp;翻译的效果看起来不是特别好。

### 4.4.3 基于OpenAI接口的机器翻译实验

&emsp;&emsp;现在来试试ChatGPT的效果。

#### 1. 简单上手版：短文本英翻中


```python
def translate_text(text):
    content = f"请将以下中文文本翻译成英文:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": content}]
    )
    translated_text = response.get("choices")[0].get("message").get("content")
    return translated_text

text_to_translate = "大家好，一起来参加DataWhale的《ChatGPT使用指南》组队学习课程吧！"
translated_text = translate_text(text_to_translate)
# 翻译文本
translated_text == "Hello everyone, let's join the team learning course of \"ChatGPT User Guide\" organized by DataWhale together!"
```

&emsp;&emsp;可以看到，ChatGPT明显比刚刚的模型效果更好，不仅语义正确，而且将《ChatGPT使用指南》翻译得更加具体。

#### 2. 进阶深度版：长文本英翻中

&emsp;&emsp;在以上所述内容中，我们更多地是了解了如何对短文本实现摘要、纠错、翻译等功能，目前ChatGPT仅支持有限个词汇输入。但是在实际场景中，特别是对于翻译问题，往往需要对很长的输入文本进行处理。一个简单的想法是，将输入文本进行切割，每次切出不超过模型能接受的最大单词数的文本进行处理，并保存结果输出，最后将所有的输出拼接到一起得到最终结果。

&emsp;&emsp;下面，我们将以翻译《哈利波特》英文原著为例，学习如何处理长文本翻译任务。

&emsp;&emsp;第一步，导入图书。

```python
with open("dataset/哈利波特1-7英文原版.txt", "r") as f:
    text = f.read()
# 全书字符数
len(text) == 6350735
```

&emsp;&emsp;整本书的字符数为635万多，但我们知道，ChatGPT的接口调用费用是根据Token数量来的，我们可以简单地使用`tokenizer`来统计所有Token数量。

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # GPT-2的tokenizer和GPT-3是一样的
token_counts = len(tokenizer.encode(text))
# 全书token数
token_counts == 1673251

# ChatGPT的API调用价格是 1000 token 0.01美元，因此可以大致计算翻译一本书的价格
translate_cost = 0.01 / 1000 * token_counts
# 翻译全书费用
translate_cost == 16.73251
```

&emsp;&emsp;这里，我们使用`GPT2Tokenizer`统计全书的Token数，并根据ChatGPT的接口调用价格来估计翻译一整本书的价格。得到翻译全书约需人民币115元，有点贵了，试着只翻译第一本。

```python
end_idx = text.find("2.Harry Potter and The Chamber Of Secrets.txt")
text = text[:end_idx]
# 第一册字符数
len(text) == 442815

tokenizer = GPT2Tokenizer.from_pretrained("gpt2") 
token_counts = len(tokenizer.encode(text))
# 第一册token数
token_counts == 119873

translate_cost = 0.01 / 1000 * token_counts
# 翻译第一册费用
translate_cost == 1.19873
```

&emsp;&emsp;只翻译第一册约需要人民币9元，相对还算实惠。

&emsp;&emsp;类似ChatGPT这样的大语言模型一般对输入Token长度有限制，因此可能无法直接将12万Token的文本全部输进去。我们可以使用一个简单的方法，将文本分成若干份，每一份使用ChatGPT翻译，最终再拼接起来。

&emsp;&emsp;当然了，随意的切割文本是不合理的，在保证每块文本长度低于最大限制长度的条件下，我们最好还能保证每份文本本身的语义连贯性。如果从一个句子中间将上下文拆成两块，则翻译时容易存在歧义。一个比较直观的想法是，将每个段落当成一个文本块，每次翻译一段。但是本书的段落非常多，有3000多段，而每段文本的单词数相对较短，最长的段落仅有275个单词。显然，一段一段翻译显然会降低翻译的效率，同时，由于每段的上下文较少，导致翻译错误的可能性上升。


```python
paragraphs = text.split("\n")
# 段落数
len(paragraphs) == 3038

ntokens = []
for paragraph in paragraphs:
    ntokens.append(len(tokenizer.encode(paragraph)))
# 最长段落的token数
max(ntokens) == 275
```


&emsp;&emsp;因此，我们选定一个阈值，如500，每次加入一个文本段落，如果总数超过500，则开启一个新的文本块。


```python
def group_paragraphs(paragraphs, ntokens, max_len=1000):
    """
    合并短段落为文本块，用于丰富上下文语境，提升文本连贯性，并提升运算效率。
    :param paragraphs: 段落集合
    :param ntokens: token数集合
    :param max_len: 最大文本块token数
    :return: 组合好的文本块
    """
    batches = []
    cur_batch = ""
    cur_tokens = 0

    # 对于每个文本段落做处理
    for paragraph, ntoken in zip(paragraphs, ntokens):
        if ntoken + cur_tokens + 1 > max_len:  # "1" 指的是"\n"
            # 如果加入这段文本，总token数超过阈值，则开启新的文本块
            batches.append(cur_batch)
            cur_batch = paragraph
            cur_tokens = ntoken
        else:
            # 否则将段落插入文本块中
            cur_batch += "\n" + paragraph
            cur_tokens += (1 + ntoken)
    batches.append(cur_batch)  # 记录最后一个文本块
    return batches

batchs = group_paragraphs(paragraphs, ntokens, max_len=500)
# 文本块数
len(batchs) == 256

new_tokens = []
for batch in batchs:
    new_tokens.append(len(tokenizer.encode(batch)))
# 最长文本块的token数
max(new_tokens) == 500
```

&emsp;&emsp;经过段落的重新组合，我们得到了256个文本块，其中最长的文本块长度为500。

&emsp;&emsp;实操中发现，由于接口使用速率限制，用ChatGPT翻译长文本很慢，这里改用`Completion`接口实现。

```python
def translate_text(text):
     content = f"请将以下英文文本翻译成中文:\n{text}"
     response = openai.ChatCompletion.create(
         model="gpt-3.5-turbo", 
         messages=[{"role": "user", "content": content}]
     )
     translated_text = response.get("choices")[0].get("message").get("content")
     return translated_text

def translate_text(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"请将以下英文翻译成中文:\n{text}",
        max_tokens=2048
    )
    translate_text = response.choices[0].text.strip()
    return translate_text
```


&emsp;&emsp;接下来，我们对每个文本块做翻译，并将结果合并起来。


```python
from tqdm import tqdm

translated_batchs = []
translated_batchs_bak = translated_batchs.copy()
cur_len = len(translated_batchs)
for i in tqdm(range(cur_len, len(batchs))):
    translated_batchs.append(translate_text(batchs[i]))
```

&emsp;&emsp;有的时候由于网络问题，可能会出现连接中断或连接超时错误。一种方法是在断点处开始重跑。另一种方法是，加入重试机制，如果失败则尝试自动重连。以下脚本会在失败后随机等待一段时间并重跑，如果重试6次仍失败，则整个任务失败。

```python
from tenacity import retry, stop_after_attempt, wait_random_exponential

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def translate_text(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"请将以下英文翻译成中文:\n{text}",
        temperature=0.3,
        max_tokens=2048
    )

    translate_text = response.choices[0].text.strip()
    return translate_text

for i in tqdm(range(len(batchs))):
    translated_batchs.append(translate_text(batchs[i]))
```

&emsp;&emsp;保存结果至txt文件，这样，我们便拥有了一份完整的翻译文件。

```python
result = "\n".join(translated_batchs)

with open("dataset/哈利波特1中文版翻译.txt", "w", encoding="utf-8") as f:
    f.write(result)
```

## 4.5 本章小结

&emsp;&emsp;在本章中，我们主要学习了ChatGPT在自然语言生成任务中的应用。首先，我们简单介绍了自然语言生成任务的一些基础知识，接着对文本摘要、文本纠错、机器翻译三个具体的任务分别进行了介绍。对于文本摘要任务，我们对比了传统方法与ChatGPT模型的输出结果，并基于`ada`模型对自定义语料进行微调。对于文本纠错任务，我们同样对比了传统方法与大模型，并基于一些工具或者自定义函数实现了输出的可视化展示。最后，对于机器翻译任务，一方面我们学习了ChatGPT在短文本翻译上的应用，另一方面通过将输入文本进行切割与组合，实现了长文本的翻译任务。


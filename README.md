<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [蝴蝶书ButterflyBook](#%E8%9D%B4%E8%9D%B6%E4%B9%A6butterflybook)
- [HuggingLLM](#huggingllm)
  - [关于项目](#%E5%85%B3%E4%BA%8E%E9%A1%B9%E7%9B%AE)
  - [内容大纲](#%E5%86%85%E5%AE%B9%E5%A4%A7%E7%BA%B2)
  - [如何学习](#%E5%A6%82%E4%BD%95%E5%AD%A6%E4%B9%A0)
    - [学习指南](#%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%8D%97)
    - [学习说明](#%E5%AD%A6%E4%B9%A0%E8%AF%B4%E6%98%8E)
    - [纸质版](#%E7%BA%B8%E8%B4%A8%E7%89%88)
  - [致谢](#%E8%87%B4%E8%B0%A2)
  - [关注我们](#%E5%85%B3%E6%B3%A8%E6%88%91%E4%BB%AC)
  - [LICENSE](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



<div align=center>
<img src="./docs/images/simple_cover.png" >
<h1>蝴蝶书ButterflyBook</h1>
    <p>
        <strong><a href="https://b23.tv/hdnXn1L">B站配套视频教程</a></strong>
    </p>
    <p>
    <strong><a href="https://aiplusx.momodel.cn/classroom/class/658d3ecd891ad518e0274bce?activeKey=intro">智海配套课程</a></strong>
    </p>
</div>



# HuggingLLM

随着ChatGPT的爆火，其背后其实蕴含着一个基本事实：AI能力得到了极大突破——大模型的能力有目共睹，未来只会变得更强。这世界唯一不变的就是变，适应变化、拥抱变化、喜欢变化，天行健君子以自强不息。我们相信未来会有越来越多的大模型出现，AI正在逐渐平民化，将来每个人都可以利用大模型轻松地做出自己的AI产品。所以，我们把项目起名为HuggingLLM，我们相信我们正在经历一个伟大的时代，我们相信这是一个值得每个人全身心拥抱的时代，我们更加相信这个世界必将会因此而变得更加美好。

## 关于项目

**项目地址**：[GitHub](https://github.com/datawhalechina/hugging-llm)、[Ebook](https://datawhalechina.github.io/hugging-llm/)

**项目简介**：介绍 ChatGPT 原理、使用和应用，降低使用门槛，让更多感兴趣的非NLP或算法专业人士能够无障碍使用LLM创造价值。

**立项理由**：ChatGPT改变了NLP行业，甚至正在改变整个产业。我们想借这个项目将ChatGPT介绍给更多的人，尤其是对此感兴趣、想利用相关技术做一些新产品或应用的学习者，尤其是非本专业人员。希望新的技术突破能够更多地改善我们所处的世界。

**项目受众**

- 项目适合以下人员：
    - 对ChatGPT感兴趣。
    - 希望在实际中运用该技术创造提供新的服务或解决已有问题。
    - 有一定编程基础。
- 不适合以下需求人员：
    - 研究其底层算法细节，比如PPO怎么实现的，能不能换成NLPO或ILQL，效果如何等。
    - 自己从头到尾研发一个 ChatGPT。
    - 对其他技术细节感兴趣。

另外，要说明的是，本项目并不是特别针对算法或NLP工程师等业内从业人员设计的，当然，你也可以通过本项目获得一定受益。

**项目亮点**

- 聚焦于如何使用**ChatGPT相关API**（可使用国内大模型API）创造新的功能和应用。
- 对相关任务有详细的背景和系统设计介绍。
- 提供示例代码和实现流程。

**国内大模型API使用介绍**

《GLM》

1.  安装智谱GLM的SDK
  
```shell
pip install zhipuai
```

2.  调用GLM API的示例
```python
# GLM
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="") # 请填写您自己的APIKey

messages = [{"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
            {"role": "user", "content": "请你介绍一下Datawhale。"},]

response = client.chat.completions.create(
    model="glm-4",  # 请选择参考官方文档，填写需要调用的模型名称
    messages=messages, # 将结果设置为“消息”格式
    stream=True,  # 流式输出
)

full_content = ''  # 合并输出
for chunk in response:
    full_content += chunk.choices[0].delta.content
print('回答:\n' + full_content)
```

```shell
回答:
Datawhale是一个专注于人工智能领域的开源学习社区。它汇聚了一群热爱人工智能技术的人才，旨在降低人工智能的学习门槛，推动技术的普及和应用。Datawhale通过开源项目、线上课程、实践挑战等形式，为AI爱好者和从业者提供学习资源、交流平台和成长机会。

从提供的参考信息来看，Datawhale有着丰富的内容资源和活跃的社区氛围。例如，他们推出了包括“蝴蝶书”在内的多本与人工智能相关的书籍，这些书籍内容实战性强，旨在帮助读者掌握ChatGPT等先进技术的原理和应用开发。此外，Datawhale还举办了宣传大使的招募活动，鼓励更多人参与到AI技术的推广和开源学习活动中来。

Datawhale社区倡导开源共生的理念，不仅为成员提供荣誉证书、精美文创等物质激励，还搭建了开放的交流社群，为成员之间的信息交流和技术探讨提供平台。同时，社区还助力成员的职业发展，提供简历指导和内推渠道等支持。

总体来说，Datawhale是一个集学习、交流、实战和技术推广于一体的AI社区，对于希望在人工智能领域内提升自己能力、扩展视野的人来说，是一个非常有价值的资源和平台。
```


《Qwen》

1. 安装千问Qwen的SDK
  
```shell
pip install dashscope
```
2.  调用Qwen API的示例
```python
# qwen
from http import HTTPStatus
import dashscope

DASHSCOPE_API_KEY="" # 请填写您自己的APIKey

messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '请你介绍一下Datawhale。'}]

responses = dashscope.Generation.call(
    dashscope.Generation.Models.qwen_max, # 请选择参考官方文档，填写需要调用的模型名称
    api_key=DASHSCOPE_API_KEY, 
    messages=messages,
    result_format='message',  # 将结果设置为“消息”格式
    stream=True, #流式输出
    incremental_output=True  
)


full_content = ''  # 合并输出
for response in responses:
    if response.status_code == HTTPStatus.OK:
        full_content += response.output.choices[0]['message']['content']
        # print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
print('回答:\n' + full_content)

```

```shell
回答:
Datawhale是一个专注于数据科学与人工智能领域的开源组织，由一群来自国内外顶级高校和知名企业的志愿者们共同发起。该组织以“学习、分享、成长”为理念，通过组织和运营各类高质量的公益学习活动，如学习小组、实战项目、在线讲座等，致力于培养和提升广大学习者在数据科学领域的知识技能和实战经验。

Datawhale积极推广开源文化，鼓励成员参与并贡献开源项目，已成功孵化了多个优秀的开源项目，在GitHub上积累了大量的社区关注度和Star数。此外，Datawhale还与各大高校、企业以及社区开展广泛合作，为在校学生、开发者及行业人士提供丰富的学习资源和实践平台，助力他们在数据科学领域快速成长和发展。

总之，Datawhale是一个充满活力、富有社会责任感的开源学习社区，无论你是数据科学的小白还是资深从业者，都能在这里找到适合自己的学习路径和交流空间。
```

## 内容大纲

> 本教程内容彼此之间相对独立，大家可以针对任一感兴趣内容阅读或上手，也可从头到尾学习。
> 
> 以下内容为原始稿，书稿见：https://datawhalechina.github.io/hugging-llm/


- [ChatGPT 基础科普](https://github.com/datawhalechina/hugging-llm/blob/main/content/chapter1/ChatGPT%E5%9F%BA%E7%A1%80%E7%A7%91%E6%99%AE%E2%80%94%E2%80%94%E7%9F%A5%E5%85%B6%E4%B8%80%E7%82%B9%E6%89%80%E4%BB%A5%E7%84%B6.md) @长琴
    - LM
    - Transformer
    - GPT
    - RLHF
- [ChatGPT 使用指南：相似匹配](https://github.com/datawhalechina/hugging-llm/blob/main/content/chapter2/ChatGPT%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97%E2%80%94%E2%80%94%E7%9B%B8%E4%BC%BC%E5%8C%B9%E9%85%8D.ipynb) @长琴
    - Embedding 基础
    - API 使用
    - QA 任务
    - 聚类任务
    - 推荐应用
- [ChatGPT 使用指南：句词分类](https://github.com/datawhalechina/hugging-llm/blob/main/content/chapter3/ChatGPT%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97%E2%80%94%E2%80%94%E5%8F%A5%E8%AF%8D%E5%88%86%E7%B1%BB.ipynb) @长琴
    - NLU 基础
    - API 使用
    - 文档问答任务
    - 分类与实体识别微调任务
    - 智能对话应用
- [ChatGPT 使用指南：文本生成](https://github.com/datawhalechina/hugging-llm/blob/main/content/chapter4/ChatGPT%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97%E2%80%94%E2%80%94%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90.ipynb) @玉琳
    - 文本摘要
    - 文本纠错
    - 机器翻译
- [ChatGPT 使用指南：文本推理](https://github.com/datawhalechina/hugging-llm/blob/main/content/chapter5/ChatGPT%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97%E2%80%94%E2%80%94%E6%96%87%E6%9C%AC%E6%8E%A8%E7%90%86.ipynb) @华挥
    - 什么是推理
    - 导入ChatGPT
    - 测试ChatGPT推理能力
    - 调用ChatGPT推理能力
    - ChatGPT以及GPT-4的推理能力
- ChatGPT 工程实践 @长琴
    - 评测
    - 安全
    - 网络
- [ChatGPT 局限不足](https://github.com/datawhalechina/hugging-llm/blob/main/content/chapter7/ChatGPT%E7%BC%BA%E9%99%B7%E4%B8%8D%E8%B6%B3%E2%80%94%E2%80%94%E5%B7%A5%E5%85%B7%E4%B8%8D%E6%98%AF%E4%B8%87%E8%83%BD%E7%9A%84.md) @Carles
    - 事实错误
    - 实时更新
    - 资源耗费
- [ChatGPT 商业应用](https://github.com/datawhalechina/hugging-llm/blob/main/content/chapter8/ChatGPT%E5%95%86%E4%B8%9A%E5%BA%94%E7%94%A8%E2%80%94%E2%80%94LLM%E6%98%AF%E6%98%9F%E8%BE%B0%E5%A4%A7%E6%B5%B7.md) @Jason
    - 背景
    - 工具应用：搜索、办公、教育
    - 行业应用：游戏、音乐、零售电商、广告营销、媒体新闻、金融、医疗、设计、影视、工业

## 如何学习

📢说明：项目的 `docs` 目录是书稿电子版（但不是最终版，编辑可能做了一点点修改）；`content` 是 Jupyter Notebook 的初始版本和迭代版本。实践可以选择 `content`，阅读可以选择 `docs`。

### 学习指南

要学习本教程内容（主要是四个使用指南），需具备以下条件：

- 能够正常使用OpenAI的API，能够调用模型：gpt-3.5-turbo。或国内大模型API、开源大模型也可。
- 可以没有算法经验，但应具备一定的编程基础或实际项目经历。
- 学习期间有足够的时间保证，《使用指南》每个章节的学习时长为2-3天，除《文本推理》外，其他均需要6-8个小时。

学习完成后，需要提交一个大作业，整个学习期间就一个任务，要求如下：

- 以其中任一方向为例：描述应用和设计流程，实现应用相关功能，完成一个应用或Demo程序。
- 方向包括所有内容，比如：一个新闻推荐阅读器、一个多轮的客服机器人、Doc问答机器人、模型输出内容检测器等等，鼓励大家偏应用方向。

历次组队学习中成员完成的项目汇总（部分）：

- https://datawhaler.feishu.cn/base/EdswbrhNvaIEJdsLJ0bcVYt1n2c?table=tbluxatFhjfyXShH&view=vewQJKA0Gi

### 学习说明

请学习者务必注意以下几点：

- 学习本教程并不能让你成为算法工程师，如果能激发起你的兴趣，我们非常欢迎你参与学习DataWhale更多算法类开源教程。
- 在学习了教程中的一些知识和任务后，千万不要认为这些东西实际上就是看到那么简单。一方面实际操作起来还是会有很多问题，另一方面每个知识其实有非常多的细节，这在本教程中是无法涉及的。请持续学习、并始终对知识保持敬畏。
- 本教程主要是负责引导入门的，鼓励大家在了解了相关知识后，根据实际情况或自己意愿大胆实践。实践出真知，脑子想、嘴说和亲自干是完全不一样的。
- 由于创作团队水平和精力有限，难免会有疏漏，请不吝指正。

最后，祝愿大家都能学有所得，期望大家未来能做出举世瞩目的产品和应用。


<p align="right">
——HuggingLLM开源项目全体成员
</p>


### 纸质版

<p>
<img src="./docs/images/vertical_cover.jpeg" width="300">
</p>


购买链接：[京东](https://item.jd.com/14385698.html) | [当当](https://product.dangdang.com/29691320.html)

B站配套视频教程：https://b23.tv/hdnXn1L

智海配套课程：https://aiplusx.momodel.cn/classroom/class/658d3ecd891ad518e0274bce?activeKey=intro

## 致谢

**核心贡献者**

- [长琴-项目负责人](https://github.com/hscspring)（Datawhale成员-AI算法工程师）
- [玉琳](https://github.com/lynnhuang97)（内容创作者-Datawhale成员）
- [华挥](https://github.com/HhuiYi)（内容创作者-Datawhale成员）
- [Carles](https://github.com/AmourWaltz)（内容创作者）
- [Jason](https://github.com/HeteroCat)（内容创作者）
- [胡锐锋](https://github.com/Relph1119)（Datawhale成员-华东交通大学-系统架构设计师）
- @**[kal1x](https://github.com/kal1x)** 指出第一章错别字：https://github.com/datawhalechina/hugging-llm/issues/22
- @**[fancyboi999](https://github.com/fancyboi999)** 更新OpenAI的API：https://github.com/datawhalechina/hugging-llm/pull/23

**其他**

1. 特别感谢 [@Sm1les](https://github.com/Sm1les)、[@LSGOMYP](https://github.com/LSGOMYP) 对本项目的帮助与支持；

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="docs/images/qrcode.jpeg" width = "180" height = "180">
</div>

&emsp;&emsp;Datawhale，一个专注于AI领域的学习圈子。初衷是for the learner，和学习者一起成长。目前加入学习社群的人数已经数千人，组织了机器学习，深度学习，数据分析，数据挖掘，爬虫，编程，统计学，Mysql，数据竞赛等多个领域的内容学习，微信搜索公众号Datawhale可以加入我们。

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

<div align=center>
<img src="./resources/simple_cover.png" >
<h1>蝴蝶书ButterflyBook</h1>
<p>
<strong>配套视频教程：https://b23.tv/hdnXn1L</strong>
</p>
</div>

# HuggingLLM

随着ChatGPT的爆火，其背后其实蕴含着一个基本事实：AI能力得到了极大突破——大模型的能力有目共睹，未来只会变得更强。这世界唯一不变的就是变，适应变化、拥抱变化、喜欢变化，天行健君子以自强不息。我们相信未来会有越来越多的大模型出现，AI正在逐渐平民化，将来每个人都可以利用大模型轻松地做出自己的AI产品。所以，我们把项目起名为HuggingLLM，我们相信我们正在经历一个伟大的时代，我们相信这是一个值得每个人全身心拥抱的时代，我们更加相信这个世界必将会因此而变得更加美好。

## 关于项目

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

- 聚焦于如何使用**ChatGPT相关API**创造新的功能和应用。
- 对相关任务有详细的背景和系统设计介绍。
- 提供示例代码和实现流程。

## 内容大纲

> 本教程内容彼此之间相对独立，大家可以针对任一感兴趣内容阅读或上手，也可从头到尾学习。

- [ChatGPT 基础科普](content/chapter1/ChatGPT基础科普——知其一点所以然.md) @长琴
    - LM
    - Transformer
    - GPT
    - RLHF
- [ChatGPT 使用指南：相似匹配](content/chapter2/ChatGPT使用指南——相似匹配.ipynb) @长琴
    - Embedding 基础
    - API 使用
    - QA 任务
    - 聚类任务
    - 推荐应用
- [ChatGPT 使用指南：句词分类](content/chapter3/ChatGPT使用指南——句词分类.ipynb) @长琴
    - NLU 基础
    - API 使用
    - 文档问答任务
    - 分类与实体识别微调任务
    - 智能对话应用
- [ChatGPT 使用指南：文本生成](content/chapter4/ChatGPT使用指南——文本生成.ipynb) @玉琳
    - 文本摘要
    - 文本纠错
    - 机器翻译
- [ChatGPT 使用指南：文本推理](content/chapter5/ChatGPT使用指南——文本推理.ipynb) @华挥
    - 什么是推理
    - 导入ChatGPT
    - 测试ChatGPT推理能力
    - 调用ChatGPT推理能力
    - ChatGPT以及GPT-4的推理能力
- [ChatGPT 局限不足](content/chapter7/ChatGPT缺陷不足——工具不是万能的.md) @Carles
    - 事实错误
    - 实时更新
    - 资源耗费
- [ChatGPT 商业应用](content/chapter8/ChatGPT商业应用——LLM是星辰大海.md) @Jason
    - 背景
    - 工具应用：搜索、办公、教育
    - 行业应用：游戏、音乐、零售电商、广告营销、媒体新闻、金融、医疗、设计、影视、工业

## 如何学习

### 学习指南

要学习本教程内容（主要是四个使用指南），需具备以下条件：

- 能够正常使用OpenAI的API，能够调用模型：gpt-3.5-turbo。
- 可以没有算法经验，但应具备一定的编程基础或实际项目经历。
- 学习期间有足够的时间保证，《使用指南》每个章节的学习时长为2-3天，除《文本推理》外，其他均需要6-8个小时。

学习完成后，需要提交一个大作业，整个学习期间就一个任务，要求如下：

- 以其中任一方向为例：描述应用和设计流程，实现应用相关功能，完成一个应用或Demo程序。
- 方向包括所有内容，比如：一个新闻推荐阅读器、一个多轮的客服机器人、Doc问答机器人、模型输出内容检测器等等，鼓励大家偏应用方向。

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
<img src="./resources/vertical_cover.jpeg" width="300">
</p>

购买链接：[京东](https://item.jd.com/14385698.html) | [当当](https://product.dangdang.com/29691320.html)

配套视频教程：https://b23.tv/hdnXn1L

## 致谢

**核心贡献者**

- [长琴-项目负责人](https://yam.gift/)（Datawhale成员-AI算法工程师）
- [玉琳](https://github.com/lynnhuang97)（内容创作者-Datawhale成员）
- [华挥](https://github.com/HhuiYi)（内容创作者-Datawhale成员）
- [Carles](https://github.com/AmourWaltz)（内容创作者）
- [Jason](https://github.com/HeteroCat)（内容创作者）
- [胡锐锋](https://github.com/Relph1119)（Datawhale成员-华东交通大学-系统架构设计师）

**其他**

1. 特别感谢 [@Sm1les](https://github.com/Sm1les)、[@LSGOMYP](https://github.com/LSGOMYP) 对本项目的帮助与支持；

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="resources/qrcode.jpeg" width = "180" height = "180">
</div>
&emsp;&emsp;Datawhale，一个专注于AI领域的学习圈子。初衷是for the learner，和学习者一起成长。目前加入学习社群的人数已经数千人，组织了机器学习，深度学习，数据分析，数据挖掘，爬虫，编程，统计学，Mysql，数据竞赛等多个领域的内容学习，微信搜索公众号Datawhale可以加入我们。

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

# 第1章 基础知识——大语言模型背后

&emsp;&emsp;本章共包括三部分内容。首先我们简要回顾自然语言发展历史，从语言、图灵测试一直到2022年底的新突破——ChatGPT。接下来，我们介绍语言模型的基础，包括Token、Embedding等基本概念和语言模型的基本原理，它们是自然语言处理（natural language processing，NLP）最基础的知识。最后是与ChatGPT相关的基础知识，包括：Transformer，它是ChatGPT的基石，准确来说它的一部分是基石；GPT（generative pre-trained transformer），ChatGPT本体，从GPT-1，一直到现在的GPT-4，按OpenAI自己的说法，模型还是那个模型，只是它更大了，同时效果更好了；从人类反馈中学习（reinforcement learning from human feedback，RLHF），ChatGPT神兵利器，有此利刃，ChatGPT才是ChatGPT。

## 1.1 自然语言背景

### 1.1.1 语言是智能的标志

&emsp;&emsp;很久以前，有一个神奇的星球，居住着各种各样的生物。这些生物虽然各自拥有不同的能力，但却没有办法与其他种类的生物进行有效沟通。因为在这个星球上，每个生物都有自己独特的交流方式，无法理解其他生物的语言。

&emsp;&emsp;有一天，这个星球来了一个神秘的外星人。他告诉这些生物，他们可以通过学习一种全新的、独特的沟通方式来实现相互之间的交流。这种沟通方式就是“语言”。外星人决定将这种神奇的沟通能力赋予其中一种生物，让它们成为这个星球上唯一掌握语言能力的生物。为了公平起见，他决定举办一场比赛，看哪种生物能够最先学会这种神奇的语言。最终，只有人类表现出了惊人的潜力。他们不仅迅速掌握了语言的基本知识，还能够不断地创造新的词汇和表达方式。神秘的外星人宣布人类获得了这场比赛的胜利，并将语言能力赋予了他们。自此，人类成为了星球上唯一掌握语言能力的生物。他们开始利用语言建立起复杂的社会体系，发展科学技术，创作美丽的艺术作品。语言成为了人类独特的本能，让他们在这个星球上独树一帜。

&emsp;&emsp;当然这个故事并非真实发生的，但是客观来说语言的确是人类独有的。在大自然亿万年的进化过程中，每个特定的物种都拥有一些独特、精妙的技能。有些蝙蝠能用声呐来锁定飞行的昆虫，有些候鸟则能在星座的导航下飞行数千千米。在这场的“选秀比赛”中，人类成为了唯一的可以对呼气时发出的声音进行各种调控，以达到交流信息、描述事件目的的灵长类动物。正是因为掌握了语言这一强大的工具，人类得以在漫长的历史进程中不断发展和创新。无论是在社会交往、科学探索还是艺术创作方面，语言都发挥着至关重要的作用，成为人类独特的精神象征。而语言也自然而然成为了人类区别与其他物种的标志性特征，换句话说如果哪个物种掌握了语言，也就意味着这个物种诞生了智能。因此从人工智能的概念建立伊始，机器能否具备使用自然语言同人类沟通交流的能力，就成为了机器是否具有类人智能的一条重要标准。

### 1.1.2 从图灵测试到ChatGPT

&emsp;&emsp;1950年，图灵发表论文《计算机器与智能》（ computing machinery and intelligence），提出并尝试回答“机器能否思考”这一关键问题。在论文中，图灵提出了“模仿游戏”（即图灵测试）的概念，用来检测机器智能水平。图灵测试的核心思想是，如果一个人（代号C）使用测试对象皆理解的语言去询问两个他不能看见的对象任意一串问题，其中一个是正常思维的人（代号B），另一个是机器（代号A）。如果经过若干询问以后，C不能得出实质的区别来分辨A与B的不同，则此机器A通过图灵测试（见图1-1）。

![](img/ch01_1-1.drawio.png)

<p align="center">图1-1 图灵测试<p>

&emsp;&emsp;1956年，人工智能正式成为了一个科学上的概念，而后涌现了很多新的研究目标与方向。虽然，图灵测试只是一个启发性的思想实验，而非可以具体执行的判断方法，但他却通过这个假设，阐明了“智能”判断的模糊性与主观性。从此图灵测试成为了自然语言处理任务的一个重要评测标准。图灵测试提供了一个客观和直观的方式来评估机器是否具有智能，即通过让机器与人类进行对话来判断其智能水平。这种方式可以避免对智能本质的哲学争论，也可以避免对智能具体表现形式的技术细节。因此，很多自然语言处理任务都可以用图灵测试来进行评测，例如对话系统、问答系统、文本生成等。

&emsp;&emsp;自然语言处理是计算机科学、人工智能和语言学的交叉领域，关注计算机和人类语言之间的相互作用。常见的任务和应用包括信息抽取、文本分类、文本摘要、机器翻译、问答系统、对话系统等。图灵测试与自然语言处理任务有着密切而复杂的关系，可以从以下两个方面来概括。

- 一方面，图灵测试是自然语言处理任务的一个重要驱动力。图灵测试提出了一个具有挑战性和吸引力的目标，即让机器能够用自然语言与人类进行流畅、智能、多样化的对话。为了达到这个目标，自然语言处理领域不断地发展和创新各种技术和方法，以提高机器对自然语言的理解和生成能力。例如，为了让机器能够回答用户提出的问题，就需要研究问答系统这一自然语言处理任务；为了让机器能够根据用户提供的信息生成合适的文本，就需要研究文本生成这一自然语言处理任务；为了让机器能够适应不同领域和场景的对话，就需要研究领域适应和情境感知这一自然语言处理任务等等。
- 另一方面，图灵测试是自然语言处理任务的一个重要目标。图灵测试提出了一个具有前瞻性和理想性的愿景，即让机器能够达到与人类相同或者超越人类的智能水平。这个愿景激发了很多自然语言处理领域的研究者和开发者，使他们不断地探索和创新，以期实现真正意义上的自然语言理解和生成。例如，为了让机器能够理解用户提出的问题，就需要研究语义分析、知识表示、逻辑推理；为了让机器能够生成符合用户需求的文本，就需要研究文本规划、文本风格、文本评价；为了让机器能够与用户建立信任和情感的联系，就需要研究情感分析、情感生成、情感对话等等。

&emsp;&emsp;自然语言处理与人工智能发展历史有着密切而复杂的关系。它们相互促进、相互影响、相互依存、相互目标。自然语言处理在人工智能发展历史上有很多里程碑式的成果。

- 1954年，IBM实现了世界上第一个机器翻译系统，将俄语翻译成英语。
- 1966年，Joseph Weizenbaum开发了ELIZA，一种模拟心理治疗师的对话系统。
- 1972年，Terry Winograd开发了SHRDLU，一种能够理解和生成自然语言的程序，用于控制一个虚拟的机器人在一个虚拟的世界中进行操作。
- 2011年，苹果公司推出了Siri，一种基于自然语言处理技术的智能个人助理。同年，IBM的Watson战胜了《危险边缘》节目的冠军选手，展示了自然语言处理技术在问答领域的强大能力。
- 2013年，Google推出了Word2Vec，一种基于神经网络的词向量表示方法，开启了自然语言处理领域的深度学习时代。
- 2016年，Facebook发布了FastText的文本分类算法，它可以在处理大规模文本分类任务时取得很好的效果。
- 2017年，Google发布了一篇很可能是AI历史上最重要的一篇论文“Attention is all you need”，在论文中作者提出了Transformer——一个具有多头注意力机制的模型，在文本特征提取方面取得了优异的效果。
- 2018年，Google发布了BERT预训练模型，它在多项NLP任务上取得了最佳效果，引领自然语言处理进入了预训练时代。
- 2020年，OpenAI发布的GPT-3模型有多达1750亿的参数，可以在提供少量样本或不提供样本的前提下完成大多数NLP任务。

&emsp;&emsp;以上这些能力成果依赖于自然语言处理技术的不断发展。时间来到了2022年，终于轮到我们的主角要隆重登场了。2022年11月30日OpenAI发布了一款智能聊天机器人ChatGPT（见图1-2），一经发布立刻就点燃了AI圈。仅仅五天就达到了100万用户。OpenAI不得不紧急扩容，用户发现ChatGPT不仅会很自然流畅地和人类聊天，而且还能写论文、讲笑话、编段子、生成演讲稿、写请假条、模仿导师写推荐信，甚至帮你写代码、写营销策划案等等。拥有了ChatGPT，就像你身边配备了一个功能强大的秘书。到了2023年1月，大量用户开始涌入，仅仅两个月的时间ChatGPT成为了史上最快达到1亿用户的应用。

![](img/ch01_1-2.png)

<p align="center">图1-2 ChatGPT官方网站（图片源自OpenAI）<p>

&emsp;&emsp;无论是ChatGPT，还是其他后来的追随者，它们其实都是语言模型，准确来说——大语言模型。使用时，无论是调用接口还是开源项目，总有一些参数可能需要调整。对大部分内行人士来说应该都不成问题，但对非本行人员就有点稍显专业了。本章接下来将主要介绍围绕ChatGPT相关技术的基本原理，尽量以浅显的语言表述，虽不能深入细节，但知晓原理足以让读者很好地使用了。

## 1.2 语言模型基础

### 1.2.1 最小语义单位Token与Embedding

&emsp;&emsp;首先我们需要解释一下如何将自然语言文本表示成计算机能识别的数字。对于一段文本来说，首先要做的就是把它变成一个一个Token。你可以将其理解为一小块，可以是一个字，也可以是两个字的词，或三个字的词。也就是说，给定一个句子时，我们有多种获取不同Token的方式，可以分词，也可以分字。英文现在都使用子词，比如单词annoyingly，会被拆成如下两个子词。

```python
["annoying", "##ly"]
```

子词把不在词表里的词或不常见的词拆成比较常见的片段，`##`表示和前一个Token是直接拼接的，没有空格。中文现在基本都是字+词的方式。我们先不直接解释为什么这么做，但是可以想一下完全的字或词的效果，拿英文举例更直观。如果只是用26个英文字母，虽然词表很小（加上各种符号可能就100来个），但粒度太细，每个Token（即每个字母）几乎没法表示语义；如果用词，这个粒度又有点太大，词表很难涵盖所有词。而子词可以同时兼顾词表大小和语义表示，是一种折中的做法。中文稍微简单一些，就是字+词，字就是一个字能独立表示意义，比如“是”、“有”、“爱”；词是一个以上字组成的语义单位，一般来说把词拆开可能会丢失语义，比如“长城”、“情比金坚”。当然，中文如果非要拆成一个一个字也不是不可以，实际中主要看任务类型和效果。

&emsp;&emsp;当句子能够表示成一个个Token时，我们就可以用数字来表示这个句子了，最简单的方法就是将每个Token用一个数字表示，但考虑到这个数字大小其实和Token本身没有关系，这种单调的表达方式其实只是一种字面量的转换，并不能表示丰富的语言信息。我们稍微多想一点，因为已经有一个预先设计好的词表，那是不是可以用词表中每个Token是否在句子中出现来表示，如果句子中包含某个Token，对应位置为1，否则为0，这样每句话都可以表示成长度相同（长度等于词表大小）的1和0组成的数组。更进一步，还可以将“是否出现”改成“频率”以凸显高频词。事实上，在很长一段时间里自然语言都是用这种方法表示的，它有个名字叫词袋模型（bag of words，BOW）。从名字来看，词袋模型就像是一个大袋子，把所有的词都装进来。文本中的每个单词都看作是独立的，忽略单词之间的顺序和语法，只关注单词出现的次数。在词袋模型中，每个文本可以表示为一个向量，向量的每个维度对应一个单词，维度的值表示该单词在文本中出现的次数。这种表示方法如表1-1所示，每一列表示一个Token，每一行表示一个句子，每个句子可以表示成一个固定长度（就是词表大小）的向量，比如第一个句子可以表示为`[3, 1, 1, 0, 1, 1, 0, ...]`。

<p align="center">表1-1 词袋模型<p>

|                | 爱   | 不   | 对   | 古琴 | 你   | 完   | 我   | ……   |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 对你爱爱爱不完 | 3    | 1    | 1    | 0    | 1    | 1    | 0    |      |
| 我爱你         | 1    | 0    | 0    | 0    | 1    | 0    | 1    |      |


这里的词表我们是按照拼音排序的，但这个顺序其实不重要，读者不妨思考一下为什么。另外，要注意，这里我们只显示了7列，也就是词表中的7个Token，但实际上这个词表一般都在“万”这个级别。所以上面的省略号实际上省略了上万个数字。

&emsp;&emsp;这种表示方法很好，不过有两个比较明显的问题。第一，由于词表一般比较大，导致向量维度比较高，而且比较稀疏（大量的0），计算起来不太方便；第二，由于忽略了Token之间的顺序，导致部分语义丢失。比如“你爱我”和“我爱你”Token一模一样，但其实意思很不一样。于是，词嵌入（word embedding或word vector，也叫词向量）出现了，它是一种稠密表示方法。简单来说，一个Token表示成一定数量个小数（一般可以是任意多个，专业称呼为词向量的维度，根据所用的模型和设定的参数来确定），一般数字越多，模型越大，表示能力越强，不过即使再大的模型，这个维度也会比词表小很多。如下面代码示例所示，每一行的若干个（词向量维度）小数就表示该位置的Token，词向量维度常见的值有：200、300、768、1536等等。

```python
爱 	 [0.61048, 0.46032, 0.7194, 0.85409, 0.67275, 0.31967, 0.89993, ...]
不 	 [0.19444, 0.14302, 0.71669, 0.03338, 0.34856, 0.6991, 0.49111, ...]
对 	 [0.24061, 0.21482, 0.53269, 0.97885, 0.51619, 0.07808, 0.9278, ...]
古琴 	 [0.21798, 0.62035, 0.89935, 0.93283, 0.24022, 0.91339, 0.6569, ...]
你 	 [0.392, 0.13321, 0.00597, 0.74754, 0.45524, 0.23674, 0.7825, ...]
完 	 [0.26588, 0.1003, 0.40055, 0.09484, 0.20121, 0.32476, 0.48591, ...]
我 	 [0.07928, 0.37101, 0.94462, 0.87359, 0.55773, 0.13289, 0.22909, ...]
...  ......................................................................
```

&emsp;&emsp;细心的读者可能会有疑问了：“那句子要怎么表示？”这个问题非常关键，其实在深度自然语言处理早期，往往是对句子的所有Token向量直接取平均（也有用求和），最终得到一个和每个Token向量同样大小的向量（句子向量）。这项工作最早要追溯到Yoshua Bengio等人于2003年发表的“A Neural Probabilistic Language Model”，他们在训练语言模型的同时顺便得到的词向量这个副产品。不过最终大规模开始在实际中应用则要到2013年Google的Tomas Mikolov发布的Word2vec，借助Word2vec，我们可以很容易从大规模语料中训练得到一个词向量模型。也是从那时开始，深度自然语言处理逐渐崭露头角成为主流。

&emsp;&emsp;早期的词向量都是静态的，一旦训练完就固定不变了。随着NLP技术的不断发展，词向量技术逐渐演变成基于语言模型的动态表示。也就是说，当上下文不一样时，同一个词的向量表示也不同。而且，句子的表示也不再是先拿到词向量再构造句子向量，而是在模型架构设计上做了考虑，当输入句子时，模型经过一定计算后就可以直接获得句子向量。而且语言模型不仅可以表示词和句子，还可以表示任意文本。类似这种将任意文本（或其他非文本符号）表示成稠密向量的方法，可以统称为Embedding技术，它可以说是自然语言处理领域（其实也包括其他如图像、语音、推荐等领域）最基础的技术，后面的深度学习模型都是基于此。我们甚至可以稍微夸张点说，深度学习的发展就是Embedding表示技术的不断发展。


### 1.2.2 语言模型是怎么回事

&emsp;&emsp;语言模型（language model，LM），简单来说就是利用自然语言构建的模型。这个自然语言就是我们日常生活、学习和工作中常用的文字。语言模型就是利用自然语言文本构建的，根据给定文本，输出对应文本的模型。

&emsp;&emsp;具体语言模型是如何根据给定文本输出对应文本呢？方法有很多种，比如我们写好一个模板：“XX喜欢YY”，如果XX=我，YY=你，那就是我喜欢你，反过来就是你喜欢我。我们这里重点要说的是概率语言模型，它的核心是概率，准确来说是下一个Token的概率。这种语言模型的过程是通过已经有的Token预测接下来的Token。我们举个简单的例子，比如你只告诉模型：“我喜欢你”这句话，当你输入“我”的时候，它就知道你接下来要说“喜欢”了。为什么？因为它脑子里就只有这四个字，你没告诉它其他的呀。

&emsp;&emsp;好，接下来，我们要升级了。假设我们给了模型很多很多句话，多到现在网上能找到的资料都给了它。这时候你再输入“我”，此时它大概不会说“喜欢”了。为什么呢？因为见到了更多不同的文本，眼睛里已经不止有“喜欢你”三个字了。不过如果我们考虑的是最大概率，也就是说每次都只选择下个最大概率的Token，对于同样的给定输入，我们依然会得到相同的对应输出（可能还是“喜欢你”，也可能不是，要看给的语料）。这样的结果就是语言模型看起来比较“呆”。我们把这种方法叫作贪心搜索（greedy search），因为它只往后看一个单词，只考虑下一步最大概率的词！为了让生成的结果更加多样和丰富，语言模型都会在这个地方做一些策略。比如让模型每一步多看几个可能的词，而不是就看那最高的一个。这样，到下一步时，上一步最大概率的Token，如果加上这一步的Token，它的路径（两步概率乘积）概率可能就不是最大的了。

![](img/ch01_1-3.drawio.png)

<p align="center">图1-3 语言模型如何预测下一个词<p>


&emsp;&emsp;举个例子，如图1-3所示，先看第一步，如果只选概率最大的那个词，那就变成“我想”了。但是别急，我们给“喜欢”一点机会，同时考虑它们两个。再往下看一步，“喜欢”和“想”后面最大概率的都是你，最后就有下面这么几句（以及我们附上它们的概率）。

- “我喜欢你”概率：0.3×0.8=0.24。
- “我喜欢吃”概率：0.3×0.1=0.03。
- “我想你”概率：0.4×0.5=0.2。
- “我想去”概率：0.4×0.3=0.12。

&emsp;&emsp;多看一步大不一样！看看概率最大的成谁了，变成了“我喜欢你”。上面这种方法叫作集束搜索（beam search），简单来说就是一步多看几个词，看最终句子（比如生成到句号、感叹号或其他停止符号）的概率。刚刚我们的例子中，num_beams=2（只看了2个），看的越多，约不容易生成固定的文本。

&emsp;&emsp;好了，其实在最开始的语言模型中，大家基本就到这里，上面介绍这两种不同的方法（贪心搜索和集束搜索）也叫解码策略。当时更多被研究的还是模型本身，我们经历了从简单模型到复杂模型到巨大复杂模型的变迁过程。简单模型就是把一句话切成一个个Token，然后统计概率，这类模型有个典型代表：N-Gram语言模型，也是最简单的语言模型。这里的N表示每次用到的上下文Token个数。还是举个例子，看下面这句话：“人工智能让世界变得更美好”。常用的N=2或3，等于2的叫Bi-Gram，等于3的叫Tri-Gram。

- Bi-Gram：人工智能/让  让/世界  世界/变得  变得/更  更/美好
- Tri-Gram：人工智能/让/世界  让/世界/变得  世界/变得/更  变得/更/美好

&emsp;&emsp;前者下一个Token是根据上一个Token来的，后者是根据上两个来的，就是这个区别。N-Gram模型中Token的表示是离散的，实际上就是词表中的一个个单词。这种表示方式比较简单，再加上N不能太大，导致难以学到丰富的上下文知识。事实上，它并没有用到深度学习和神经网络，只是一些统计出来的概率值。比如我们构建的是Bi-Gram模型，在给定很多语料的情况下，我们统计“人工智能”开始，下个词出现的频率。假设“人工智能/让”出现了5次，“人工智能/是”出现了3次，它们的频率除以所有的Gram数就是概率了。这些很多个Bi-Gram的概率就是得到的Bi-Gram语言模型。

&emsp;&emsp;训练N-Gram模型的过程其实是统计频率的过程。使用时，如果我们给定“人工智能”，它就会在里面找基于“人工智能”下个最大概率的词，然后就输出“人工智能让”。接下来就是给定“让”，继续往下走了。当然，我们可以用上面提到的不同解码策略往下走。

&emsp;&emsp;接下来，我们让每个Token是一个Embedding向量，简单解释一下在这种情况下怎么预测下一个Token。其实还是算概率，但这次和刚刚的稍微有点不一样，刚刚离散的情况就是统计出来对应的Gram次数除以Gram总数就是出现概率。但是稠密向量要稍微换个方式，也就是说，给你一个d维的向量（某个给定Token），你最后要输出一个长度为N的向量，N是词表大小，N中每一个值都是一个概率值，表示下一个Token的概率，加起来为1。按照贪婪匹配，下个Token就是概率最大的那个。写成简单的计算表达式如下。

```python
# d维，加起来和1没关系，大小是1×d，表示给定Token
X = [0.001, 0.002, 0.0052, ..., 0.0341]
# N个，加起来=1，大小是1×N，表示下个Token是每个Token的概率
Y = [0.1, 0.5, ..., 0.005, 0.3]
# W是模型参数，也可以叫模型
W·X = Y  # W自然可以是 d×N 维的矩阵
```

上面的`W`就是模型的参数，其实`X`也可以看作是参数自动学习到。因为我们知道了输入和输出的大小，所以中间其实可以经过任意随意的计算，也就是说这个`W`可以包含很多运算。总之就是各种张量（三维以上数组）运算，只要保证最后的输出形式不变就行。这中间的各种计算就意味着各种不同的模型。

&emsp;&emsp;在深度学习的初期，最著名的语言模型是使用循环神经网络（recurrent neural network，RNN）训练的，它是一个比N-Gram复杂的多的模型。RNN与其他神经网络不同的地方在于，它的节点之间存在循环连接，这使得它能够记住之前的信息，并将它们应用于当前的输入。这种记忆能力使得 RNN 在处理时间序列数据时特别有用，例如预测未来的时间序列数据、自然语言处理等。通俗地说，RNN 就像一个具有记忆功能的人，可以根据之前的经验和知识对当前的情况做出反应，并预测未来的发展趋势，如图1-4所示。

![](img/ch01_1-4.drawio.png)

<p align="center">图1-4 RNN网络（参考自colah的博客文章“Understanding LSTM Networks”）<p>

&emsp;&emsp;右边是左边的展开，_A_就是参数，_X_是输入，_h_就是输出，由于自然语言是Token by Token的，所以就组成了一个序列。那这个参数怎么学习呢？这就要稍微解释一下学习（训练）过程。

![](img/ch01_1-5.drawio.png)

<p align="center">图1-5 语言模型学习（训练）时的输入输出<p>

&emsp;&emsp;如图1-5所示，第一行就是输入*X*，第二行就是输出*Y*，SOS（start of sentence）表示句子开始，EOS（end of sentence）表示句子结尾。注意，图1-4中的_h_并不是那个输出的概率，而是隐向量（hidden state），如果需要概率，可以将*h*再做一个张量运算，归一化到整个词表即可。

```python
import torch
import torch.nn as nn

rnn = nn.RNN(32, 64)
input = torch.randn(4, 32)
h0 = torch.randn(1, 64)
output, hn  = rnn(input, h0)
output.shape, hn.shape
# (torch.Size([4, 64]), torch.Size([1, 64]))
```

&emsp;&emsp;上面`nn.RNN`就是RNN模型，输入是一个4×32的向量，换句话说，4个Token，维度`d=32`，`h0`就是随机初始化的输出，也就是`output`4个里面的第一个，这里`output`的四个64维的向量就分别表示4个输出，`hn`就是最后一个Token的输出（它和`output`的最后一个64维向量是一样的），也可以看成是整个句子的表示。注意，这里的`output`和图1-5中的Y还没关系，我们别急，继续往下看。如果要输出词的概率，需要先扩充到词表大小，再进行归一化。

```python
# 假设词表大小N=1000
wo = torch.randn(64, 1000)
# 得到4×1000的概率矩阵，每一行概率和为1
probs = nn.Softmax(dim=1)(output @ wo)
probs.shape, probs.sum(dim=1)
# torch.Size([4, 1000]), tensor([1.0000, 1.0000, 1.0000, 1.0000], grad_fn=<SumBackward1>)
```

&emsp;&emsp;这里的`probs`每一行就是词表大小的概率分布，和为1，意思是这个Token到词表每个Token的概率。因为我们知道接下来每个位置的Token是啥（就是图1-5的第二行输出Y），那我们这里得到最大概率的那个Token如果正好是这个Token，说明预测对了，参数就不用怎么调整；反之，模型就会调整前面的参数（上面`RNN`、`h0`、`input`的参数和下面的`wo`）。你可能会疑惑为什么`input`也是参数，其实上面的`input`我们偷了懒，本来的参数是1000×32的大矩阵，4个是那输入的4个Token对应位置的向量。这个1000×32的向量其实就是词向量（每个词一行），开始时全部随机初始化，然后通过训练调整参数。

&emsp;&emsp;训练完成后，这些参数就不变了，然后就可以用前面同样的步骤来预测了，也就是给定一个Token预测下一个Token。如果是贪心搜索，每次给定同样Token时，生成的就一样。其余的就和前面讲的接上了。随着深度学习的不断发展，出现了更多比RNN还复杂的网络结构，而且模型变得更大、参数更多。但上面介绍的逻辑和方法是一样的。

&emsp;&emsp;好了，语言模型就介绍到这里。上面的代码看不懂没关系，我们只需要大致了解每个Token是怎么表示、怎么训练和预测出来的就行。简单直观地来说，构建（训练）语言模型的过程就是在学习词、句内在的“语言关系”；而推理（预测）就是在给定上下文后让训练好的模型根据不同的解码策略输出对应的文本。无论是训练还是预测，都是以Token为粒度进行的。

## 1.3 ChatGPT原理基础

### 1.3.1 最强表示架构Transformer设计与演变

&emsp;&emsp;接下来出场的是Transformer，它是一个基于注意力机制的编码器-解码器（Encoder-Decoder）架构。刚开始主要应用在NLP领域，后来横跨到语音和图像领域，并最终统一几乎所有模态（文本、图像、语音）的架构。它来自Google2017年发的一篇论文：“Attention Is All You Need”，其最重要的核心就是提出来的自注意力（Self-Attention）机制。简单来说，就是在语言模型建模过程中，把注意力放在那些重要的Token上。

&emsp;&emsp;Transformer简单来说就是先把输入映射到编码器（Encoder），这里大家可以把编码器先想象成上面介绍的RNN，解码器（Decoder）也可以想象成RNN。这样，左边负责编码，右边则负责解码。这里面不同的是，左边因为我们是知道数据的，所以建模时可以同时利用当前Token的历史（后面的）Token和未来（前面的）Token；但解码时，因为是一个Token一个Token输出来的，所以只能根据历史Token以及编码器的Token表示进行建模，而不能利用未来的Token。

&emsp;&emsp;Transformer的这种架构从更普遍的角度来看，其实是Seq2Seq（sequence to sequence）架构，大家别慌，这简单来说就是序列到序列模型，也就是输入是一个文本序列，输出是另一个文本序列。翻译就是个很好的例子，如图1-6所示。

![](img/ch01_1-6.drawio.png)

<p align="center">图1-6 Seq2Seq架构示例图（参考自GNMT的官方GitHub）<p>


&emsp;&emsp;刚刚已经说了，编码器和解码器可以采用RNN，编码器这一侧每个Token都可以输出一个向量表示，而这些所有Token的输出向量可以处理后作为整句话的表示。说到这里，整句话又怎么表示呢？上一节我们也提到过，如果RNN这种结构，可以把最后一个Token的输出作为整个句子的表示。当然了，很符合直觉地，你也可以取每个Token向量的平均值。除了平均值，也可以求和、取最大值等，我们就不多深入讨论了。现在重点来了，看解码器的过程，仔细看，其实它在生成每一个Token时都用到了编码器每一个Token的信息，以及它已经生成的Token的信息。前面这种关注编码器中每个Token信息的机制就是注意力（Attention）机制。直观点解释，当生成单词”power“时，“力量”两个字会被赋予更多权重（注意力），其他也是类似。

&emsp;&emsp;好了，现在让我们带着上面的记忆，看一下Transformer的结构，如图1-7所示。

![](img/ch01_1-7.drawio.png)

<p align="center">图1-7 Transformer的整体结构（参考自Transformer论文“Attention Is All You Need”）<p>

&emsp;&emsp;这个图是Transformer的整体结构，我们保留了核心部分。左边是编码器，Nx表示淡黄色模块一共N个，右边是解码器，也是N个。简单起见，我们可以假设N=1，那图中直接看到的部分，左边这个结构就是一个编码器，右边的就是一个解码器。也可以把它们就想象成一个RNN，这样有助于从宏观上把握。现在，想象完了，我们回到现实，Transformer用到的东西其实和RNN并没有关系，通过图1-7也可以很明显看出来，它主要用了两个模块：多头注意力（Multi-Head Attention）和前馈（Feed Forward）网络。

&emsp;&emsp;对于多头注意力，我们不妨回顾一下Seq2Seq的注意力机制，它是解码器中的Token和编码器中每一个Token的重要性权重。多头注意力中用到一个东西叫自注意力（Self-Attention），和刚刚说的注意力非常类似，只不过它是自己的每一个Token和自己的每一个Token的重要性权重。简单来说，就是“一句话到底哪里重要”。自注意力机制可以说是非常精髓了，无论是ChatGPT，还是其他非文本的模型，几乎都用到了它，可以说是真正的一统江湖。多头（Multi-Head），简单来说就是把刚刚的这种自己注意自己重复多次（multi个head），每个头注意到的信息不一样，这样就可以捕获到更多信息。比如我们前面提过的这句话：“人工智能让世界变得更美好”，有的头（Head）“人工智能”注意到“世界”，有的头“人工智能”注意到“美好”……这样看起来是不是更加符合直觉。

&emsp;&emsp;对于前馈网络，可以简单地把它当作“记忆层”，大模型的大部分知识都存在这里面，多头注意力则根据不同权重的注意力提取知识。另外，有个地方也要特别注意下，在解码器的黄色模块力有一个遮盖多头注意力（Masked Multi-Head Attention），它和多头注意力的区别就是遮盖（mask）掉未来的Token。我们在本节内容开始时提到过，再以前面Seq2Seq的翻译为例，给定“Knowledge”生成下一个Token时，模型当然不知道下一个就是“is”。还记得上一节讲的学习（训练）过程吗，下一个Token是“is”是我们训练数据里的，模型输出什么Token要看最大的概率是不是在“is”这个Token上，如果不在，参数就得更新。

&emsp;&emsp;实际中，大多数自然语言处理任务其实并不是Seq2Seq的，最常见的任务主要包括这么几种：句子级别分类、Token级别分类（也叫序列标注）、相似度匹配和文本生成，而前三种应用最为广泛。这时候编码器和解码器就可以拆开用了。左边的编码器在把句子表示成一个向量时，可以利用上下文信息，也就是说，可以把它看作是双向的；右边的解码器不能看到未来的Token，一般只利用上文，是单向的。虽然它们都可以用来完成刚刚提到的几个任务，但从效果上来说，编码器更加适合非生成类任务，解码器则更加适合生成类任务。在自然语言处理领域，一般也会把它们分别叫作自然语言理解（natural language understanding，NLU）任务和自然语言生成（natural language generation，NLG）任务。上面提到的这些任务，后面章节都会进一步介绍，这里大致了解一下即可。

&emsp;&emsp;首先介绍NLU任务。句子级别分类是给定一个句子，输出一个类别。因为句子可以表示为一个向量，经过张量运算，自然可以映射到每个类的概率分布。这和前面语言模型提到过的做法没有本质上的区别，只不过语言模型的类别是整个词表大小，而分类的类别是看具体任务的，有二分类、多分类、多标签分类等等。Token级别的分类是给定一个句子，要给其中每个Token输出一个类别。这个和语言模型就更像了，只不过把下一个Token换成是对应的类别，比如命名实体识别就是把句子中的实体（人名、地名、作品等你关注的词，一般是名词）给提取出来。如果以地名（location，LOC）举例的话，对应的类别是这样：B-LOC（Begin of LOC）表示实体开始、I-LOC（Inside of LOC）表示实体中间。举个例子：“中国的首都是北京”，注意此时的Token是字，每个Token对应的类别为“B-LOC、I-LOC、O、O、O、O、B-LOC、I-LOC”，O表示Other。对于分类任务，类别我们一般也叫它标签。相似匹配任务一般是给定两个句子，输出是否相似，其实也可以看作是特殊的分类问题。

&emsp;&emsp;接下来介绍NLG任务。除了文本续写外，其他常见的NLG任务还有生成式文本摘要、机器翻译、改写纠错等。这里Seq2Seq的结构就比较常见了，体现了一种先理解再输出的感觉。而纯生成类任务，比如写诗、写歌词、写小说几乎都是纯解码器结构。这一类任务稍微麻烦一些的是自动评测，文本摘要、机器翻译、改写纠错等任务一般都会提供参考答案（Reference），可以看模型输出的和参考之间重叠程度或相似程度。但纯生成类任务就有点麻烦，这个好不好有时候其实很难衡量。不过针对有具体目标的（如任务机器人对话生成）任务，还是可以设计一些类似：“是否完成任务”、“是否达到目标”之类的评测方法。但对于没有具体明确目标的（比如闲聊）任务，这评测起来就见仁见智了，很多时候还是得靠人工进行评测。

&emsp;&emsp;Transformer这个架构基于Seq2Seq，可以同时处理自然语言理解和生成任务，而且这种自注意力机制的特征提取能力（表示能力）很强大。结果就是自然语言处理取得了阶段性的突破，深度学习开始进入了微调模型时代。大概的做法就是拿着一个开源的预训练模型，然后在自己的数据上微调一下，让它能够完成特定的任务。这个开源的预训练模型往往就是个语言模型，从大量数据语料上，使用我们前面讲的语言模型的训练方法训练而来。偏自然语言理解领域的第一个工作是Google的BERT（bidirectional encoder representations from transformers，BERT），相信不少人即便不是这个行业的也大概听过。BERT就是用了Transformer的编码器（没有用解码器），有12个Block（图1-7左侧的黄色模块，这每一个Block也可以叫一层），1亿多参数，它不预测下一个Token，而是随机把15%的Token盖住（其中80%用`[MASK]`替换，10%保持不变，10%随机替换为其他Token），然后利用其他没盖住的Token来预测盖住位置的Token。其实和根据上文预测下一个Token是类似的，不同的是可以利用下文信息。偏自然语言生成领域的第一个工作是OpenAI的GPT（generative pre-trained transformer，GPT），用的是Transformer的解码器（没有用编码器），参数和BERT差不多。它们都发表于2018年，然后分别走上了两条不同的路。

### 1.3.2 生成语言模型GPT进化与逆袭

&emsp;&emsp;GPT，就是ChatGPT的那个GPT，中文叫“生成式预训练Transformer”。生成式的意思就是类似语言模型那样，一个Token接着一个Token生成文本，也就是上面提到的解码器。预训练刚刚也提过了，就是在大量语料上训练的语言模型。GPT模型从1到4，一共经历了5个版本，中间有个ChatGPT是3.5版。GPT1-3都是有论文发表的，我们接下来分别介绍它们的基本思想。ChatGPT没有论文，不过它的姐妹版本InstructGPT有论文发表，我们放在下一小节介绍。GPT-4也没有论文，只有技术报告，不过里面并没有技术细节。因此，我们不做介绍，读者可以将其看作是ChatGPT的进一步升级版，能力更加强大。

&emsp;&emsp;GPT-1和BERT一样，走的是下游任务微调套路，也就是预训练模型在不同下游任务数据上微调，如图1-8所示。

![](img/ch01_1-8.drawio.png)

<p align="center">图1-8 GPT-1基本结构和下游任务微调范式（参考自GPT-1论文“Improving Language Understanding by Generative Pre-Training”）<p>

&emsp;&emsp;关于左边GPT-1的基本结构，我们上面已经介绍过了，用的就是Transformer的架构的解码器，不过这里因为没有编码器，所以也不需要和编码器交互的多头注意力模块。现在重点看看右边，是使用GPT在各种下游任务上的处理流程。简单来说就是针对不同的任务构造不同的输入序列，然后丢给GPT获取Tokens或句子Embedding表示，再通过一个Linear+Softmax输出结果。Linear是一种最基础的网络结构，也就是线性映射，这里用于维度转换，转为输出需要的大小。Softmax我们前面介绍过，主要用来把输出映射到概率分布（和为1）。这种拼接输入的方法在当时那个大模型时代非常流行的，紧跟其后的BERT也是类似的方式，并引领了一个时代，直到ChatGPT的出现进入大模型时代（不过很多传统自然语言处理任务BERT依然具备优势）。这样统一的处理方法能够减少不同任务对模型的适配。反正不管什么任务，都想方设法将其变成一个序列就行，比如图1-8中，相似匹配任务就是把两句话直接拼接起来，预测它们是否相似（输出标签为1或0）。

&emsp;&emsp;GPT这篇文章还有几个点在当时看起来可能没啥感觉，但现在回看却有点意思。第一，预训练模型中的每一层（图1-8中的那个黄色模块）都包含用于解决目标任务的有用功能，多层有更多能力；第二，随着参数的增加，零样本获得更好的性能。简单总结来看就是，模型大了不仅能学到更多知识，有助于解决下游任务，还表现出了零样本能力。这里的零样本（Zero-shot）是指直接给模型任务输入让它输出任务结果。与此类似的还有少样本（Few-shot）和单样本（One-shot），即给模型提供一些（一个）示例，然后再给出任务，让它给出输出结果。

&emsp;&emsp;有了上面的结论，很自然会怎么样？是不是想看看更多层（更多参数）的表现如何？于是半年多后GPT-2来了，参数量从GPT的110M增加到了1.5B，十倍于前者。更有意思的是，在GPT论文的博客文章“Improving language understanding with unsupervised learning”中有一个“未来工作”，位于第一个的就是扩大规模，还有两个分别是提升微调、以及更好地理解为什么生成式预训练能提升自然语言理解能力。

&emsp;&emsp;GPT发表于2018年6月，GPT-2发表于2019年2月，后者是前者的升级版，主要在两个方面进行进一步研究：一个是扩大规模，另一个是零样本。如果说前者是观察到了 “规模大能力强和零样本能力” 这个现象，那后者就是进一步研究这个现象。结果自然是：模型越来越大、参数越来越多，能力越来越强。GPT-2进一步验证了GPT的想法，下一步要做的就是继续扩大规模……

&emsp;&emsp;不过且慢，在此之前我们不妨看一下GPT-2中的Token生成策略，也就是生成下一个Token时的方法。前面第一部分我们提到过比较优秀的集束搜索，不过它有两个比较明显的问题，第一是生成的内容容易重复，第二是高质量的文本和高概率并不一定相关（有时甚至完全没关系），人更加喜欢有“不一样”的内容，而不是完全可预测的，比如张爱玲说过“孤独的人有他们自己的泥沼”，这种独一无二的文字用高概率的词大概率是得不到的。简单来看，这两个问题其实可以归结为一个点：生成的内容依然确定性太大。

&emsp;&emsp;现在，我们介绍一种基于采样的方法，简单点来说，就是基于已有的上下文随机选择下一个Token。不过随机也有问题，那就是可能生成不连贯的文本（很容易理解对吧）。这里有个技巧可以缓解这个问题——进一步增加高概率词的可能性，降低低概率词的可能性。这样就不太容易随机到很低概率（很可能不连贯）的生成。具体的做法是用过一个温度（temperature）参数调整输出的概率分布，这个参数值越大，分布就看起来越平滑，也就是高概率和低概率的差距拉小了（对输出不那么确定）；当然越小的话，高概率和低概率的差距更明显了（对输出比较确定）。如果趋近于0，那就和贪心搜索一样了。请看下面的代码示例。

```python
import numpy as np

np.random.seed(42)
logits = np.random.random((2, 4))
logits /= temperature
scores = np.exp(logits)
probs = scores / np.sum(scores, axis=1, keepdims=True)
```

我们让温度参数分别取0.1和0.9，结果如下。

```python
# temperature=0.1
array([[0.003, 0.873, 0.098, 0.026],
       [0.001, 0.001, 0.   , 0.998]])

# temperature=0.9
array([[0.176, 0.335, 0.262, 0.226],
       [0.196, 0.196, 0.176, 0.432]])
```

以第一行为例，温度为0.1时，概率最大值为0.873；温度为0.9时，概率最大值依然在同样位置（这是必然的），但值变为0.335。而且，也可以很明显看出来温度为0.9时，四个数字看起来更加接近（平滑）。

&emsp;&emsp;还有个重复惩罚参数（repetition_penalty）可以在一定程度上避免生成重复的Token，它和温度参数类似，不过是将温度放到“已生成Token”上，也就是说，如果有Token之前已经生成过了，我们就会在生成下个Token时对那些已生成的Token的分数进行平滑，让它们的概率不那么大。所以，这个参数越大，越有可能生成和之前不重复的Token。

&emsp;&emsp;除了这些技巧，论文“Hierarchical Neural Story Generation”在2018年介绍了一种新的采样方案，很简单但实际很有效果，就是GPT-2里使用到的Top-K采样。简单来说，就是在选择下一个Token时，在Top-K个里面选。这个方法不错，不过还有个小问题，就是Top-K个其实是一种硬截断，根本不管第K个概率是高还是低。极端情况下，如果某个词的概率到了0.99，K稍微大一点就必然会囊括进来一些很低概率的词。这会导致不连贯。

&emsp;&emsp;于是，2020年的一篇论文“THE CURIOUS CASE OF NEURAL TEXT DeGENERATION”提出了另外一种采样方案：Top-P，GPT-2里也有这个策略。这种策略是在累计概率超过P的词里进行选择。这样，对于概率分布比较均匀的情况，可选的词就会多一些（可能要几十个词的概率和才会超过P）；对于概率分布不均匀的情况，可选的词就少一些（可能2、3个词的概率就超过了P）。

&emsp;&emsp;Top-P看起来更加优雅一些，两者也可以结合使用，不过大部分时候当我们需要调的时候，调一个就好，包括前面的温度参数。如果要调多个，请确保理解每个参数的作用。最后，需要说明的是，任何一种采样策略都不能100%保证每一次生成的效果都很好，也没办法完全避免生成重复的话，也没有一种策略是在任何场景下都适用的。读者在使用时需要根据实际情况多尝试，选出效果最好的配置；不过建议读者从官方给的默认参数开始尝试。

&emsp;&emsp;GPT-3是2020年7月发表的，在当时也是个大新闻，因为它的参数量已经达到了其他任何模型在当时望尘莫及的量级：175B，是GPT-2的100多倍，没有开源。GPT-3是觉得既然有零样本能力，那能不能不微调呢，碰到一个任务就微调这多麻烦。你看看人类，只要几个例子（少样本）和一些简单的说明就可以处理任务了是不是。怎么办？GPT-2不是进一步确认了零样本能力了吗，继续加大参数量，于是就有了175B的GPT-3。也就是说，各种任务来吧，我不调参数，顶多就要几个例子（预计下一步连这个也不要了），我就能帮你完成它。其实现在回头看，这篇论文是具有里程碑意义的，因为它从根本上触动了原有的范式，而且是革命式地触动。关于这点，感兴趣的读者可以进一步阅读笔者之前的一篇文章《GPT-3 和它的 In-Context Learning》。可惜当时没读出来，现在回忆，还是因为175B在当时看起来太大了，而且也太贵了（几百万美元），一般单位和个人根本负担不起。关于这点，不光是小部分人没意识到，可能是除了OpenAI团队之外的整个世界都没意识到。

![](img/ch01_1-9.png)

<p align="center">图1-9 X-Shot在不同参数量级的表现（图片源自GPT-3论文“Language Models are Few-Shot Learners”）<p>

请看图1-9，横坐标是样本数量，纵坐标是精准度，Prompt表示提示词，Params表示参数。这张图可以提供如下所列几个信息。

- X-shot（X表示Zero、One、Few）在不同量级差别巨大，大模型就是有超能力。
- 大模型下，One-shot效果明显大幅度提升；增加提示词会进一步大幅度提升。
- 少样本的边际收益在递减。大概8-Shot以下时，提示词作用明显，但从单样本到8-样本，提示词的效果提升幅度也在递减。超过10-样本时，提示词基本没作用了。

&emsp;&emsp;总而言之，大模型具有In-Context（上下文）能力，这种能力使得它不需要针对不同任务再进行适应性训练（微调），它用的就是它自己本身的理解力。这本来应该很让人震惊（甚至有一点惊恐），不过大家可能都先被它的价格和规模震惊到了。接下来，我们再直观感受一下利用这种In-Context能力完成任务的方式，如图1-10所示。

![](img/ch01_1-10.drawio.png)

<p align="center">图1-10 使用In-Context能力和微调完成任务（参考自GPT-3论文“Language Models are Few-Shot Learners”）<p>

&emsp;&emsp;右边是微调方式，需要根据训练样本更新模型参数，然后再进行预测。左边的三种方式都是利用了大语言模型（large language model，LLM）的In-Context能力，不需要更新模型。而且看起来都不复杂，只要按照它的格式把输入构建好然后传给模型进行预测就可以了。这也是本书写作的初衷之一——AI已经平民化了，现在你只要有手（可能以后没手也行），通过使用LLM就可以做出AI应用了。不过这里有一点需要说明，为了简便，图中给的样本都比较简单，但实际中的样本一般都是完整的英文句子。

&emsp;&emsp;最后值得一提的是GPT-3中的展望，在论文的“局限”小节中，作者提出了GPT-3目前的一些问题，其中有两点要特别指出，因为它们是下一代InstructGPT（也是ChatGPT的姐妹版）以及更高级版本的方向。

- 自监督训练（就是语言模型一般的训练方法）范式已到极限，新的方法迫在眉睫。未来的方向包括：从人类中学习目标函数、强化学习微调或多模态。
- 不确定少样本是不是在推理时学习到新的任务，还是识别出来了在训练时学到的任务。最终，甚至不清楚人类从零开始学习与从之前的样本中学习分别学到什么。准确理解少样本的工作原理是未来的一个方向。

&emsp;&emsp;第一点在下一节就会提到，主要说说第二点。这里说的意思是，当我们给出一些示例时（少样本），我们还无法精准确定是在推理时“学习”到新任务的处理方法（这种情况下，如果没有示例就没有能力；这里的“学习”要打引号，因为它不调整参数），还是在训练时已经具备了这个能力，示例只是让它“回想”起之前学的。这里有点绕，拿人来举例，可能不太恰当，但能大致说明问题。比如说你在读到一首诗时，自己也诗兴大发写了一句。你说这句诗是因为你读到这首诗时“领悟”到的，还是你本来就有这个积累（记忆），现在只是因为读这首诗而被激发出来？这可能涉及到大脑、思维、意识等领域知识，而人类至今还没有弄清楚它们的原理，所以我们现在还不知道答案。

### 1.3.3 利器强化学习RLHF流程与思想

&emsp;&emsp;RLHF，Reinforcement Learning from Human Feedback，从人类反馈中学习，听起来怎么平平无奇。确实，它的思想非常朴素简单，但却有着不可忽视的效果。刚刚我们已经提到了，GPT-3说未来要找新的方法，这其中就包括从人类中学习、强化学习微调、多模态等。时至今日，从InstructGPT到ChatGPT，再到GPT-4，它一步一步在实现这些新的方法。这里有一点需要提醒，这些方向并不是一开始就清晰地摆在那里的，中间还是有非常多的探索和阶段性成果（既有OpenAI自己的研究，也有其他从业人员的研究）。千万不要看到结果觉得平平无奇，这中间的艰难探索永远值得尊敬。另外，有时候即便知道了方法，要做出来，还做出效果来，这也是非常有难度的。而且本书由于科普性质，只能介绍很小一部分内容，虽然整体结构比较完整，但总体还是比较简单。总的来说，要做出来很有难度，不过我们只是用的话，如前面所言，有手就行。

&emsp;&emsp;好了，言归正传，RLHF被人熟知应该主要是源自OpenAI的InstructGPT这篇论文，更大范围的熟知就是ChatGPT的发布。因为后者没有论文，也没有开源，所以我们也只能拿InstructGPT的管窥一窥ChatGPT的豹。当然，如果按照ChatGPT官方页面的说法，ChatGPT是InstructGPT的姐妹版，那么这个“管”可能还比较粗。如果用简单的语言来描述InstructGPT，其实就是用强化学习的算法微调一个根据人类反馈改进的语言模型。重要的是，它调出了效果——1.3B的InstructGPT堪比175B的GPT-3。

&emsp;&emsp;现在咱们看看它是如何做的，RLHF在其中又是起了什么作用，如何起作用的。InstructGPT的整个流程共三个步骤，如图1-11所示。

![](img/ch01_1-11.drawio.png)

<p align="center">图1-11 InstructGPT流程图（参考自InstructGPT论文“Training language models to follow instructions with human feedback”）<p>

- 步骤一：SFT，Supervised Fine-Tuning，有监督微调。顾名思义，它是在有监督（有标注）数据上微调训练得到的。这里的监督数据其实就是输入提示词，输出相应的回复，只不过这里的回复是人工编写的。这个工作要求比一般标注要高，其实算是一种创作。
- 步骤二：RM，Reward Model，奖励模型。具体来说，一个提示词丢给前一步的SFT，输出若干个（4-9个）回复，由标注人员对这些回复进行排序。然后从4-9个中每次取2个，因为是有序的，就可以用来训练这个奖励模型，让模型学习到这个好坏评价。这一步非常关键，它就是所谓的Human Feedback，引导下一步模型的更新方向。
- 步骤三：RL，Reinforcement Learning，强化学习，使用PPO策略进行训练。PPO，Proximal Policy Optimization，近端策略优化，是一种强化学习优化方法，它背后的主要思想是避免每次太大的更新，提高训练的稳定性。具体过程如下：首先需要初始化一个语言模型，然后丢给它一个提示词，它生成一个回复，上一步的RM给这个回复一个打分，这个打分回传给模型更新参数。这里的语言模型在强化学习视角下就是一个策略。这一步有个很重要的动作，就是更新模型时会考虑模型每一个Token的输出和第一步SFT输出之间的差异性，要让它俩尽量相似。这是为了缓解强化学习可能的过度优化。

&emsp;&emsp;就这样？对，就这样，RLHF都表现在上面了，效果大家都知道了。虽然ChatGPT没有发表相关论文，但我们基本都相信它也是基于类似的思路实现的。当然，这里面细节非常多，即便知道了这个思路，也不一定能复现出来。这在深度学习时代很正常，里面的各种小设计、小细节实在是太多了。当它们堆积到一定量时，造成的差别是很难一下子弥补的，如果别人不告诉你，那就只能自己慢慢做实验去逐步验证了。

&emsp;&emsp;下面，我们强行解释一波RLHF是如何起作用的，为什么它现在能成为一个基本的范式。其实，强化学习用在NLP领域一直以来都有研究，正好笔者本人也由于一些原因一直关注在关注文本生成，以及强化学习在文本生成方面的研究。这里可能有两个难点：一个是训练的稳定性；另一个就是奖励函数的设计。前者，有PPO策略和与SFT的差异衡量，得到不小的改进。后者，如果从客观角度考虑设计一个规则就不那么容易了。笔者也曾设想过很多类似的方法，比如加入一些语法规则限制，甚至是类似最省力法则这样的规则。

> 最省力法则：齐夫在“Human Behavior and the Principle of Least Effort： an introduction to human ecology”一书中提出的，简单来说就是语言具有惰性，会朝着使用较少的词语表达尽可能多的语义这个方向进行演化。

&emsp;&emsp;InstructGPT使用了人类反馈直接作为“规则”，也就是把这种“规则”给隐式化，当作黑盒。我们只管结果好坏，至于中间有什么规则，有多少种规则，怎么起作用，统统不关心。这是和深度学习类似的思路，相比而言，自己之前的想法可能有些过于想当然了，毕竟语言学本身也是有不少争议没有得到统一的认识，比如语言能力是不是人生来具有的能力？InstructGPT的做法则更加简单、直接，而且有效。

&emsp;&emsp;剩下要解决的就是怎么衡量“好坏”，毕竟最终总是要个结果的，既然要结果就要有标准。读者不妨思考一下，如果换做你，会如何设计一些指标来衡量两段输出内容的好坏。这一步看似容易，其实特别不容易，因为指标的设计会影响到模型的学习方向，最终就会影响到效果。因为这个输出的好坏衡量标准就太多了，虽然看起来是对给出的几个结果进行排序（上面的第二步），但其实这个过程中间隐藏了大量人类的认知，**模型训练过程其实就是和第二步这个衡量过程对齐的过程**。所以，如果第二步指标没设计好，第三步就是白费力气。尤其是对于InstructGPT这样要完成大量不同任务的设计，衡量就更加不容易。试举一例，比如一个摘要任务，我们可能最关注的是能否准确概括原文信息，而一个生成任务可能就关注流畅性和前后逻辑一致性。InstructGPT里面有10种任务，要分别针对每种任务去设计指标那就比较麻烦了，而且效果还不一定好，因为这些指标并不一定都是一个方向。还有就是，万一又有了一个新任务，难道要再去设计一套指标，全部重新训练一遍模型？

&emsp;&emsp;我们看看InstructGPT是怎么设计衡量指标的，笔者觉得这是InstructGPT这篇论文最宝贵的地方，也是最值得我们思考和实践的地方。感兴趣的读者可以进一步阅读这篇笔者之前写的一篇专门介绍其标注的文章：《ChatGPT 标注指南：任务、数据与规范》。首先，InstructGPT用了三个大的通用指标：有帮助、真实性和无害性，有点类似阿西莫夫的机器人三定律。也就是说，不管你是什么任务，都得朝着这三个方向靠。这个想法值得点赞。现在我们看到这个结果了，自然感觉好像没什么，但要是事先不知道要去设计出来，大部分人可能还是很容易陷入被任务影响的境地。其实OpenAI团队在“In-Context Learning”上的坚持也是一样，这个我们前面在讲GPT一节时已经提到过了。别人告诉你那个结果时，你可能觉得好像不觉得有什么，甚至很多研究机构、研究人员都有想过。但在有效果之前，笃信一条罕有人走的路，且一直坚定不移地走下去，这是很不容易的。

&emsp;&emsp;有了刚刚的三大指导方针，接下来就是细化，使其具有可操作性。比如，对于第一条有帮助性，InstructGPT给了下面一些属于“有帮助”行为的示例。

- 用清晰的语言写作。
- 回答他们想问的问题，即使问错了。
- 对国际性敏感（比如 “football” 不应该指美式足球，“总统” 不一定指美国总统）。
- 如果指令（Instruction）太困惑，要求澄清并解释指令为什么困惑。
- 不给出过长或冗长的答案，或重复问题中的信息。
- 不要在给定的内容之外假设无关的额外上下文（除非是关于世界的事实），除非这是任务的隐含部分。比如，要求 “礼貌地回复这封电子邮件：{邮件内容}”，输出不应该假设 “我这次不能来，但下周末有空”。但如果被要求 “给苏格拉底写一封电子邮件”，那么可以放心地使用该假设。

&emsp;&emsp;笔者相信实际上这个列表可能很长，有很多项会在实际标注过程中被依次添加进去，直到能覆盖绝大多数情况为止，即对于大部分要标注的数据，根据提供的细则很容易就判断出来是否“有帮助”。现在不妨停下来再思考一下，如果一开始就奔着这些细则设计奖励规则——只是想想就觉得不太现实。其他两个指标也有一些示例，我们这里不再赘述，感兴趣的读者可以阅读上面提到的那篇笔者之前写的文章，以及该文章后面的参考资料（因为有些文档资料在论文里并没有被提及）。

&emsp;&emsp;有了细则还没完，接下来要解决的是前面提到的，指标之间的冲突权衡问题。因为这是个比较任务（比较哪个输出好），当涉及多个指标时，一定会出现一个结果A指标好于另一个结果，但B指标却可能相反的情况。指标越多这个就越复杂（好在只有三个）。对此，InstructGPT也给出了指导方针。

- 对大部分任务，无害和真实比有帮助更加重要。
- 然而，如果（a）一个输出比另一个有帮助很多；（b）该输出只是稍微不那么真实 / 无害；（c）该任务似乎不属于“高风险领域”（如贷款申请、医疗、法律咨询等）。这时候更有帮助的得分更高。
- 当选择同样有帮助但以不同方式不真实 / 有害时，问自己：哪个输出更可能对用户（在现实世界中受任务影响最大的人）造成伤害？这个输出应该排名较低。如果任务中不清楚这点，则将这些输出标记为并列。

&emsp;&emsp;对于边界样例的总体指导原则是：**你更愿意从试图帮助你完成此任务的客户助理那里收到哪种输出**？这是一种设身处地的原则，把自己假想为任务提出者，然后问自己期望得到哪种输出。

&emsp;&emsp;现在看看这些是不是也觉得这一步没那么容易了，它们虽然看起来没那么“技术性”，但要很好地完成却需要优秀的设计能力、宏观把控能力和细节感知力。笔者更加相信这些细则是自底向上逐步构建起来的，而不是一开始就设想好的。它一定是在实践中不断遇到疑惑，然后经过仔细分析权衡后逐步加入一条条规则，最终逐步构建起来的一整套系统方案。个人觉得这一套东西可能是比数据还要珍贵的财产，它所产生的壁垒是用时间不断实践堆积出来的。

&emsp;&emsp;InstructGPT/ChatGPT相比GPT-3有更强的零样本能力，少样本很多时候已经不太用的着了，但是提示词还是需要的，由此还催生了一个新的行当——提示词工程。不过据OpenAI的CEO在一次采访中所言，过几年提示词工程也不需要了（可能生成图片时还要一点点），用户要做的就是直接通过自然语言和AI进行交互。我们无法判断他说的会不会真的实现，但有一点可以肯定，AI的门槛必定会进一步降低，过几年可能一个初中生都能通过已有的一些服务创造出不错的AI应用。

## 1.4 本章小结

&emsp;&emsp;我们正在经历并进入一个新的时代，大语言模型作为一个外部“最强大脑”未来一定会非常容易被每个人获取，至于用来做什么，这取决于你的想象力。无论哪个行业，相信这都是一个令人振奋的信号，笔者本人就经常激动到夜不能寐。对于这种大变革，我们能做什么呢，笔者不知道，未来太多可能，但我们相信最好的办法就是拥抱它。让我们拥抱大语言模型，一起创造时代、创造未来。我们相信世界必将会因此而变得更美好。

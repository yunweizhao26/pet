Pre-trained language models, fine-tuned with task-specific heads, are the backbone of applied NLP, and bigger and bigger language models are coming. With this in mind, alternative methods are emerging to compete with the classifier heads used in BERT, UniLM and GPT. In particular, GPT-3 has popularized prompts, natural language inputs designed to steer the pre-trained language model itself into solving the task, rather than a classifier built on top of it. 

Prompts are interesting because they allow a practitioner to give information to the model, although in a very different fashion from standard ML supervision. In our NAACL 2021 paper, we investigate prompt-based fine-tuning, a promising alternative fine-tuning approach, and find that prompts often yield an edge over the standard approach. As we interpret a prompt as additional human-crafted information for the model, we measure that edge in terms of data points and quantify: **how many data points is a prompt worth?** 

## Prompting

In order to adapt pre-trained language models to a task, the main method is to replace the final token prediction layer of the original model with a randomly initialized linear classifier head. Supervised task data is then used to train the modified model via backpropagation, learning weights for this new head but also modifying weights deeper in the model. In this work, we call this a _head_ model. 

A competing approach is _prompting_: a broad class of methods that attempt to use the initial language model to answer the task by predicting words correlated with the classes instead of a class label. This allows them to perform classification while preserving the language model functionality. For this, _prompts_ are used: input sequences designed to produce the desired answer as textual output. 

Although this may sound abstract, this is a very natural way to reason about text for humans in practice: school exercises, for example, tend to be presented as a text input (for example, an article about Mars) and a question ("Is there life on Mars?") with an expected answer in natural text ("No"<sup>1</sup>) that maps to one of the classes of the task (presumably here, "No" to `False` and "Yes" to `True`). In this paradigm, task-specific data is presented to the model much like a grammar exercise where a student would need to fill in blanks in a fixed way over a list of sequences. Prompting attempts to use the pre-training information contained in the language model explicitly, rather than implicitly through hidden representations that get fed into the linear classifier head.  


Here's an example for SuperGLUE task BoolQ, which provides a text <span style="color: #0c593d">passage</span> and a <span style="color: #031154">question</span> and expects a boolean yes-or-no answer. This data is combined with a <span style="color: #910713">**pattern**</span> into a sequence with a single <span style="color: #ba9004">**masked token**</span> that the model must predict. This prediction is turned into a classification prediction with a pre-set *verbalizer*, a mapping between tokens and classes: the model probabilities on this token for *yes* and *no* are compared, with the final prediction being `True` if *yes* dominates and `False` if *no* does.

![image](mockups/boolqpatterns.png)

## Fine-tuning

With this, we have turned our general language model into a task-specific classifier. These language model classifiers based on prompts have been used in very diverse ways:  
 
- The preserved language modeling functionality from the pre-trained model allows them to perform without additional data, as opposed to linear classifier _heads_ that are initialized from scratch and always start at random performance. A variety of papers have used this for zero-shot classification.  
- In order to incorporate supervised task data, they can use backpropagation with the usual language modeling cross-entropy loss objective: the verbalizer token associated with the correct class then serves as the correct token prediction. This is a component of PET, and is the objective used by T5 - although T5 uses prefixes to indicate the task rather than describing it with a natural-language prompt.  
- They can also use _priming_, where the sequence that needs to be filled in is prefixed with a list of correctly-filled examples. No backpropagation is used, and the weights of the language model are never modified: instead, it can attend to correct examples at inference time. This is the method used by GPT3.  
- Finally, PET uses prompt models to pseudo-label unlabeled data that is then fed to a linear head model.  
 
In this paper, our goal is to present the fairest comparison possible with head models, so we fine-tune with backpropagation.

## How many data points is a prompt worth?

As we have seen, both heads and prompting can be used in a task specific supervised setting. The core difference is that the prompted model is given a specific sentence that roughly describes the task in addition to supervised examples. In some sense, this sentence is supervision as it tells the model about the task, but it is qualitatively a very different form of supervision than is standard in ML. How should we think about this supervision? How do we quantify how “zero-shot” this setup really is?  

We do this by comparing the _head_ and _prompt_ setups on the SuperGLUE tasks and MNLI. For each task, we extract subsets of the dataset of growing size, and repeat fine-tuning on `RoBERTa-large` with both methods on every subset, keeping everything else the same. For fairness, we tune the hyperparameters on the head baseline until they've attained the level of performance of the BERT++ baseline from the SuperGLUE leaderboard, and keep them the same for the _prompt_ model. 

The curves of final performance (on each task's metric) vs dataset size are plotted below for each task <sup>2</sup>. They allow us to contrast the amount of data required to attain a certain level of performance with both setups on a given task. We call this difference the _data advantage_ of a training setup over the other at that level of performance. We call the range of performance that has been attained by both models the _comparison window_. By integrating over it we get the _average data advantage_ of a method over the other on the task. Graphically, that is simply the area between the curves, divided by the height of the comparison window. <sup>3</sup>  

![image](mockups/advantage.png)

Here's a recapitulative table of the average data advantage of the prompt model over the head model per task, with error bounds obtained by a bootstrapping approach where we hold out one of the 4 head runs and 4 prompt runs (16 combinations total for every data size), and compute the standard deviation of those outcomes. Results are very different from task to task; they even vary for the same task on different dataset, for example for MNLI and RTE, both entailment tasks. However, on every task but WiC <sup>4</sup>, the prompt method has a significant edge. **The additional information provided by the prompt is consistently equivalent to hundreds of data points**.  

|                | MNLI     | BoolQ  | CB   | COPA    | MultiRC<sup>5</sup> | RTE    | WiC     | WSC     |
|----------------|----------|--------|------|---------|----------|--------|---------|---------|
| Prompt vs Head | 3506±536 | 752±46 | 90±2 | 288±242 | 384±378  | 282±34 | -424±74 | 281±137 |


## Patterns and verbalizers

#### Control verbalizers

Prompting has for now mostly been used as a tool for zero-shot classification, which is a natural use case. However, zero-shot is usually tricky and requires perfectly aligning the prompt and verbalizer. We have already shown that prompting could be applied more generally, including in the full-data regime. In order to contrast the zero-shot and adaptive natures of prompts, we consider a _null verbalizer_, a control with a verbalizer that is completely decorrelated from the task. For tasks that only require filling in one token (thus excluding the more free-form COPA and WSC), we replace the verbalizers, for example, "yes", "no", "maybe", "right" or "wrong",  with random first names. This makes the model unusable without training data, much like a head model. We plot the corresponding curves and perform the same advantage analysis below:

![image](mockups/nullverbalizer.png)

|                | MNLI     | BoolQ  | CB   | MultiRC<sup>4</sup> | RTE    | WiC     |
|----------------|----------|--------|------|----------|--------|---------|
| Prompt vs Head | 3506±536 | 752±46 | 90±2 | 384±378  | 282±34 | -424±74 |
| Prompt vs Null | 150±252  | 299±81 | 78±2 | 74±56    | 404±68 | -354±166 |
| Null vs Head   | 3355±612 | 453±90 | 12±1 | 309±320  | -122±62 | -70±160 |

Results are noisier than for the straight prompt vs head comparison; however, we find that even with a null verbalizer, the language model is able to adapt to the task, generally catching up with the proper prompted model even with a few data points, and generally doing either on par with or better than the head model, showing the inductive bias of the prompt patterns is beneficial even without an informative verbalizer.  

#### Influence of the pattern choice

Another choice that can make or break zero-shot classification is that of the pattern, and we investigate whether that still holds in our setting. In all of our experiments, we have re-used the pattern choices from PET - two or three quite different formulations per task - and repeated all of our prompt experiments with every pattern available on the task. We plot results below; they show that the choice of prompt does not have a significant influence, being always within random seed variance.  

![image](mockups/prompts.png)

## Mot de la fin

In this work, we investigate alternate methods of fine-tuning based on natural language prompts, that aim to use the language modeling ability of pre-trained models explicitly through word predictions, instead of implicitly through linear classifiers based on the model's internal representations. We isolate the problem of fine-tuning prompt-based classifier language models with backpropagation, and find that they generally outperform standard fine-tuned linear classifiers. We estimate this advantage in terms of data point to measure the additional information provided by the human via the prompt, and find that **writing a prompt is consistently worth hundreds of data points**. Furthermore, this advantage holds even with non-informative target tokens and is fairly robust to the choice of prompt. 

For practitioners, we believe that prompt-based fine-tuning should become a standard tool: especially for small- and middle-size task-specific datasets, designing a prompt yourself is a small effort for a sizable data advantage. For researchers, we believe that a lot of questions remain unexplored in this space: Why is the same prompt worth 3500 MNLI data points but only 282 RTE data points? How are prompts related to standard ML supervision? Do they react differently to adversarial or out-of domain examples, since they have some zero-shot behaviour?

<sup>1</sup>: Or at least not that we know of.

<sup>2</sup>: A sharp-eyed reader will have noticed that all those curves are monotonous. We've performed 4 runs for every experiment (i.e. every data size of every task for head and prompt models). For clarity, and because fine-tuning can sometimes fail for both methods, resulting in negative outliers, we report for every data size the maximum performance that has been attained at this data size or smaller, which we call the _accumulated maximum_ aggregate. This does not have a big impact on the reported data advantage besides reducing variance, and the graphical interpretation would still hold even with non-monotonous curves. 

<sup>3</sup>: We treat each metric linearly to calculate advantage; alternatively, we could re-parameterize the y axis for each task. This choice does not have a consistent effect for or against prompting. For example, emphasizing gains close to convergence increases prompting advantage on CB and MNLI but decreases it on COPA or BoolQ. 

<sup>4</sup>: where, interestingly, PET had already found prompting to be ineffective

<sup>5</sup>: The comparison window of MultiRC is too small as the head baseline fails to learn beyond majority class; we use the full region for a lower-bound result.
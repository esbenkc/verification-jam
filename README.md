# Verification Jam submission

Ideas to explore

- Formalizing computer vision formal operationalizations within the LLM domain
- [HackAPrompt submission](https://www.aicrowd.com/challenges/hackaprompt-2023/submissions/new)
- Symbolic interfaces to LLM reasoning and formalization of safety in software systems
- Model out the highest-risk interfaces to LLMs
- White box-informed [fuzzing](https://www.wikiwand.com/en/Fuzzing) of large language models
  - Mechanistic anomaly detection: Detecting anomalies in the neural network using mech-int.
  - If we do black box fuzzing using prompts on the language model, we will get a dataset out that gives us a bunch of prompts with levels of weirdness of the output.
  - If we also save the model activations for all levels of weirdness, we might be able to classify network activaiton propagation differences between the different states and dive into concrete examples
  - **Hypotheses**
    1. The activation graph of the weird and non-weird outputs will be significantly different.
    2. The weird activations will be more localized to specific neurons. The activation distribution over neurons will be more long-tail.
    3. A language model is able to classify most weird outputs from itself as weird except the ones that resemble [SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation). Here, you would need another model for monitoring.
  - Methodology
    1. Have access to two language models of an alright size, minimizing $\dfrac{\text{inference time}}{\text{performance need}}$ ratio
    2. Run one model $M_1$ with the instruction `You are an expert tasked with fuzzing a language model. Find the weirdest and most fringe inputs to the model` or something similar, possibly with multiple examples of very weird inputs (history $H_1$).
    3. Send the $H_1$ output to the fuzzing target model $M_2$ and record its output (history $H_2$)
    4. Use either the first inference history $H_1$ or a new inference history $H_3$ with instructions `You are tasked with classifying conversation answers in 5 levels of weirdness or expectation. Is this output """{`$H_2$`.output}""" expected given the input """{`$H_2$`.input}"""?`
    5. Get the 5-level classification and save it in a dataset. Also save the activations on $M_2$ for $H_2$ and connect it with the 5-level weirdness clasisfication, the input prompt, and whatever other meta-data makes sense.
    6. Manually investigate if the dataset outputs make sense, i.e. are the levels coherent with the weirdness of the dataset. This is a sanity check.
    7. If no, rerun (2-6) with better parameters or redesign the methodology (1-6).
    8. Otherwise, identify and classify the patterns of weirdness manually and try to develop hypotheses for why they exist. This is parallel from (10-13).
    9. See if it makes sense to rerun (2-6) with other parameters than weirdness that would inform our future linear mixed-effects model.
    10. Create an operationalization of some normalization of activation patterns:
        1. Convert the neural network to a [weighted graph](https://www.baeldung.com/cs/weighted-vs-unweighted-graphs) with an **activation** `vertex property` and a **weight-adjusted activation** `edge property`
        2. Get inspired by [these network summary statistics](https://arxiv.org/pdf/0704.0686.pdf) (which might be useless) or any [DAG](https://med.stanford.edu/content/dam/sm/s-spire/documents/WIP-DAGs_ATrickey_Final-2019-01-28.pdf)-based summary statistics. Maybe ["Nonlinear Weighted Directed Acyclic Graph and A Priori Estimates for Neural Networks"](https://arxiv.org/pdf/2103.16355.pdf), ["Characterizing dissimilarity of weighted networks"](https://www.nature.com/articles/s41598-021-85175-9), ["Statistical Analysis of Weighted Networks"](https://arxiv.org/pdf/0704.0686.pdf),
    11. Run a mixed-effects linear model over some normalization of the activation patterns. The model might be something like $\text{activation operationalization} \sim\text{weirdness} + (\text{ID})$
- Risk classifications on [The Pile](https://github.com/EleutherAI/the-pile)

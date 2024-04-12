### W1

#### Regarding missing Baselines

-   Please understand that [1,2] are papers on arxiv. Upon the reviewer’s request, we have conducted experiments on comparing CoLLM and E4SRec with A-LLMRec. We report the performance of CoLLM and E4SRec in the below table.
-   As shown in this [link](https://shorturl.at/myOWX), A-LLMRec outperforms E4SRec and CoLLM. Although both models adapt CF-RecSys to extract item embeddings for LLM and employ LoRA on LLM, A-LLMRec’s alignment module matches and delivers collaborative/textual knowledge to the LLM more effectively (2.83 times faster than CoLLM and 1.63 times faster than E4SRec in training).

#### Regarding the number of negative instances

-   As detailed in Appendix C, we disabled the truncation mode for the LLM and removed the maximum length constraint, allowing for the inclusion of as many candidate items as possible. However, we had to limit the number of candidate items to 20 due to the memory constraint of GPUs (A6000 48VRAM).
-   However, we fully agree with the reviewer that such a constraint raises concerns about realism. Please refer to our answer to Reviewer Jv7e - Q4 for an alternative approach to avoid the memory constraint.

#### Regarding the number of negative instances

-   As detailed in Appendix C, we disabled the truncation mode for the LLM and removed the maximum length constraint, allowing for the inclusion of as many candidate items as possible. However, we had to limit the number of candidate items to 20 due to the memory constraint of GPUs (A6000 48VRAM).
-   However, we fully agree with the reviewer that such a constraint raises concerns about realism. Please refer to our answer to Reviewer Jv7e - Q4 for an alternative approach to avoid the memory constraint.

### W3

-   We apologize for any confusion we may have caused regarding the usage of the term “scenario.” We will replace “scenario” with “setting” throughout the paper to clearly deliver our experimental settings.

### W4

-   We agree with the reviewer that CTRL does not explicitly tackle the cold-start problem, although the paper mentions cold-start problems in existing collaborative models. We will make sure to correct this in the revised manuscript. Moreover, we will precisely mention that MoRec conducts experiments on cold-start items. Also, following Reviewer Uiu4's feedback, we plan to add a cold-start recommendation model as a baseline. We thank the reviewer for carefully reviewing our paper!

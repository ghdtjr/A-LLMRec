### W2.1

-   Following reviewer’s suggestion, we also compared A-LLMRec with additional baselines. However, we excluded Dropoutnet [1] from our comparison for fairness, as it relies on user content features, which are not utilized in our study. Additionally, for fair comparisons, CLCRec [2] and ALDI [3] use the same textual data as A-LLMRec for their item content features. The comparison results with CLCRec and ALDI on general and cold-item scenarios are presented in the table[link](https://shorturl.at/bfrTW).

### W2.2

-   Please refer to our answer to Reviewer 1G7b W2-3.

### W3

-   We conduct parameter studies on the effect of embedding dimension to the LLM prompting [link](https://shorturl.at/aikNV). Parameter studies on other hyper-parameters are reported in Appendix D and Table 12.
    (d: embedding dimension of CF-RecSys, and d’: output embedding dimension of the encoders (f^enc_I, f^enc_T) and input dimension of F_I)
-   A-LLMRec demonstrates robustness over the size of the embedding dimension. This demonstrates that A-LLMRec can perform effectively on recommendation tasks without the need for carefully conducted tuning of the embedding dimension size.

### W4

-   Upon the reviewer’s request, we conducted the efficiency study including SASRec which is the representative baseline in traditional CF models. The overall training time is 7.19 minutes (32.33 times faster than A-LLMRec, 81.84 times faster than TALLRec) and the inference time for all the testset is 81.29 seconds.

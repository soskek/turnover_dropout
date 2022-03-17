# Turnover Dropout

This is a simple pytorch implementation of [turn-over dropout](https://arxiv.org/abs/2012.04207) (Kobayashi et al. 2020).
The method can estimate learning influence of a training instance on another instance via dropout using instance-specific masks and their flipped masks.


### Example

```
pip install .
python run_data_cleansing.py
```



### Citation

```
@inproceedings{kobayashi2020efficient,
    title = "Efficient Estimation of Influence of a Training Instance",
    author = "Kobayashi, Sosuke and Yokoi, Sho and Suzuki, Jun and Inui, Kentaro",
    booktitle = "Proceedings of SustaiNLP: Workshop on Simple and Efficient Natural Language Processing",
    month = nov,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2012.04207",
    doi = "10.18653/v1/2020.sustainlp-1.6",
    pages = "41--47"
}
```
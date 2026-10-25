| 数据集 | 范式 | EM | EM(relaxed) | F1 | 平均步数 |
|---|---|---|---|---|---|
| 2WIKI | Baseline | 6.0 | — | 15.6 | 1.0 |
| 2WIKI | ReAct | 6.0 | 6.0 | 7.6 | 3.31 |
| 2WIKI | Plan-and-Execute | 7.0 | 12.0 | 11.9 | 2.73 |
| 2WIKI | Self-Ask | 9.0 | 9.0 | 12.4 | 2.69 |
| 2WIKI | IRCoT | 13.0 | 13.0 | 17.1 | 1.13 |
| MUSIQUE | Baseline | 1.0 | — | 7.0 | 1.0 |
| MUSIQUE | ReAct | 1.0 | 2.0 | 7.1 | 3.44 |
| MUSIQUE | Plan-and-Execute | 3.0 | 3.0 | 6.3 | 2.82 |
| MUSIQUE | Self-Ask | 2.0 | 4.0 | 9.9 | 2.70 |
| MUSIQUE | IRCoT | 1.0 | 2.0 | 10.0 | 1.05 |

注:泛化实验:2WikiMultihopQA / MuSiQue dev 前 100 题,索引取各自前 500 题语料,max_steps=8

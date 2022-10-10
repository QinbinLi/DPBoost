This is the code of paper [Privacy-Preserving Gradient Boosting Decision Trees](https://arxiv.org/pdf/1911.04209.pdf), a joint work with [Zhaomin Wu](https://github.com/JerryLife), [Zeyi Wen](https://zeyiwen.github.io/), and [Bingsheng He](https://www.comp.nus.edu.sg/~hebs/). The implementation is based on [LightGBM](https://github.com/microsoft/LightGBM).


## Script
Instructions to run DPBoost:
```
cd python-package
python3 setup.py install --user
cd ..
python3 run_exp.py
```

## Parameters
In function ```try_DPBoost_2level``` of run_exp.py:

*output_path*: The output file path.
 
*n_trees*: Number of trees.

*total_budgets*: The privacy budget

*inner_boost_round*: Number of trees inside an ensemble.

## Note
1. Since we directly implement the code based on LightGBM, it may overwrite the vanilla LightGBM of your python library.

2. The master branch is for regression task. For the binary classification task, please use the code of binary-classification branch [here](https://github.com/QinbinLi/DPBoost/tree/binary-classification).

## Contact
Please contact me by email liqinbin1998@gmail.com or create issues if you have any question.


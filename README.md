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
 
*n_trees_list*: The list of total number of trees you want to try.

*total_budgets_list*: The list of budget you want to try.

*inner_boost_round_list*: The list of number of trees in an ensemble you want to try.

## Note
1. Since we directly implement the code based on LightGBM, it may overwrite the vanilla LightGBM of your python library.

2. Currently the code only supports setting ```objective``` as ```regression``` (use square loss function). 
For the binary classification task, we convert it to the regression task (e.g., for class in [-1,1], the output class is 0 if the prediction score is bigger than 0).

## Contact
Please contact me by email liqinbin1998@gmail.com or create issues if you have any question.


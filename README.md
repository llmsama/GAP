## Requirements

All the code are based on Python 3 in Anaconda.

###Install gurobi

```setup
conda install -c gurobi gurobi
```

If you cannot install by this command, see <https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-> for help.
### Gurobi license
You need to get a license for gurobi (see <https://www.gurobi.com/academia/academic-program-and-licenses/>)

# GAP Algorithm
Code for data generation of different distribution and GAP algorithm

```Data-generation.py```

includes the generation of different datasets, including Bernoulli distribution, Uniform distribution and Truncated Normal distribution.

```Algorithm.py ```

includes the codes for GAP algorithm， Greedy algorithm as well as LP algorithm

```run_epsilon.py ```
runs the GAP algorithm with regard to different values of epsilon

```run_T.py ```
runs the GAP algorithm with regard to different values of arrival sequence length T

```run_k.py```
runs the GAP algortihm with regard to different values of bin capacity k.

```run_gamma.py```
runs the GAP algortihm with regard to different values of gamma.

```run_rou.py```
runs the GAP algortihm with regard to different values of rou.

```run_T0.py```
runs the GAP algortihm with regard to different values of empty run length T0.

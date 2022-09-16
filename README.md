# Code for reproducing results of publication [Redacted]

This repository provides all the necessary code to reproduce simulations and
figures presented in the work [title redacted for integrity of the double blind
review process].

## Scripts

All inputs to the scripts can be inspected by passing the command line argument
`-h` or `--help`. `simulate_comp.py` and `empirical_scaling.py` produce results
in JSON format while the rest of the scripts deal with plotting the results.

### `simulate_comp.py`

Runs an agent-based simulation with the given subjectivity parameters and
simulates the comparative and majority-vote method method with the same rater
population. Records the f1 scores and biases for different numbers of
comparisons in STDOUT encoded in JSON format.

### `empirical_scaling.py`

Receives as input the empirical crowdsourced comparative task results in CSV
format, and a number of items. Selects a ransom subset of all items presented in
the data and calculates f1 scores for different numbers of comparisons and
returns them in STDOUT encoded in JSON format.


### `plot_heatmap.py`

Receives a list of `.json` files containing results of `simulate_comp.py` where
each file contains a unique combination of seed value and a selected parameter,
e.g. personal threshold variance, and produces a score heatmap similar to Fig 3.
Files with the same value for the parameter, but with different seeds are
averaged to produce means.

### `plot_scores_lines.py`

Receives a list of `.json` files containing results of `simulate_comp.py` where
each file contains a unique combination of seed value and a selected parameter,
e.g. personal threshold variance. As opposed to `plot_heatmap.py`, this one
produces a set of lines for each value of the parameter, similar to Fig 4.


Files with the same value for the parameter, but with different seeds are
averaged to produce means. The standard error of mean is used for producing the
error bars.


### `plot_bias_lines.py`

Similar to `plot_scores_lines.py` but always plots the biases for different
values of the parameter beta. Produces a figure similar to Fig 5.

### `plot_votes.py`

Similar to `plot_scores_lines.py` but always plots the one trajectory for the
comparative method, but different points for the majority-vote method with
different numbers of votes. Produces a figure similar to Fig 6.

### `plot_scaling.py`

Reads JSON files from harcoded path `output/simulations/scaling/{size}/*.json`
where size takes values of 2048, 4096, 8192, 16384 and 32768. Each JSON file
should contain the output of `simulate_comp.py` with unique seed value.

Also reads JSON files from hardcoded path
`output/empirical/scaling/{size}/*.json` with sizes 25, 30, 35 and 40. Each file
should be the output of `empirical_scaling.py` with a unique combination of
shuffling and selection seeds.

This script produces a four-panel figure similar to Fig 7.

## Data

All generated data used in producing the results is located in a tarball and
gzipped file in the root of the repository titled `output.tar.gz`. The empirical
study results in CSV form is available in form of the file `pilot1+2.json` which
can be consumed directly by the script `empirical_scaling.py`.

## Reproduction instructions

You can simply use the pre-computed results in file `output.tar.gz` to reproduce
just the figures. For reproduction of the results, make sure the correct
version (including minor and tiny/patch) of the packages are installed in your
environment, then reproduce each file using either `simulate_comps.py` (for
files in `output/simulation`) or `empirical_scaling.py` (for file in
`output/empirical`) based on the parameters of that file. The parameters used to
produce each file is encoded in the file itself, under the key "params".

The results are produced using (Anaconda) Python version 3.9.13, NumPy 1.19.5
and SciPy 1.8.1 on an Intel(R) Xeon(R) CPU E5-2680 CPU running CentOS Linux
7.9.2009 (Core).

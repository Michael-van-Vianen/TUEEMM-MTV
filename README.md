# Local Subgroup Discovery on Attributed Network Graphs
## About
This Code accompanies the paper "A Multi-Target Generalization of Local Subgroup Discovery on Attributed Network Data".

The authors are Carl Vico Heinrich, Tommie Lombarts, Jules Mallens, Luc Tortike, David Wolf, and Wouter Duivesteijn of the Eindhoven University of Technology. Corresponding author is Wouter Duivesteijn who can be reached via `w.duivesteijn@tue.nl`.

## Python Environment
* The code has been tested with the Python version `3.12.9`.
* The dependencies are listed in the `requirements.txt` file and can be run with `pip install -r requirements.txt`.

## License
The code is licensed under MIT. Please see the `LICENSE.txt` file for more information.

## Datasets
The code uses the following datasets:
* [Twitch PT](https://arxiv.org/abs/1909.13021) imported from [Torch Geometric Datasets](https://pytorch-geometric.readthedocs.io/en/2.5.3/modules/datasets.html)
* [Amazon Computers](https://arxiv.org/abs/1811.05868) imported from [Torch Geometric Datasets](https://pytorch-geometric.readthedocs.io/en/2.5.3/modules/datasets.html)

## Running the Code
Depending on the dataset, the code has to be run differently.
* For the **Twitch PT** dataset, run vertex_appoach, and edge_approach to generate the top100 subgroups csv's for the respective dataset. This canbe done using 'edges.csv', 'features.json', 'target.csv'
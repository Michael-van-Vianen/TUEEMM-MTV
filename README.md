# Local Subgroup Discovery on Attributed Network Graphs
## About
This Code accompanies the paper "Local Subgroup Discovery on Attributed Network Graphs" which is part of the IDA2025 conference. The conference proceedings are published in Springers Lecture Notes in Computer Science (LNCS).

The authors are Carl Vico Heinrich, Tommie Lombarts, Jules Mallens, Luc Tortike, David Wolf, and Wouter Duivesteijn of the Eindhoven University of Technology. Corresponding author is Wouter Duivesteijn who can be reached via `w.duivesteijn@tue.nl`.

## Python Environment
* The code has been tested with the Python version `3.12.9`.
* The dependencies are listed in the `requirements.txt` file and can be run with `pip install -r requirements.txt`.

## License
The code is licensed under MIT. Please see the LICENSE file for more information.

## Datasets
The code uses the following datasets:
* OGBG-MolHIV (Primary Dataset)
* Twitch PT
* WebKB Cornell

## Running the Code
Depending on the dataset, the code has to be run differently.
* For the OGBG-MolHIV dataset, one first has to run `OGBG-MolHIV_inspecting.ipynb` to inspect the dataset and to generate the `.pkl` file which is needed by `OGBG-MolHIV_dataset.ipynb`. After that, one can run `OGBG-MolHIV_dataset.ipynb` which runs the prposed alogrithm on the dataset and does the ablation study. It also generates a `.csv` file with the results of the algorithm which is needed as input for the `OGBG-MolHIV_visualization.ipynb` file which visualizes the results. Inside each of the files one has to specify the protein number which one wants to inspect, the protein number has to match within all three files.
* For the Twitch PT and the WebKB Cornell dataset one first has to run the `{dataset name}_dataset.ipynb` file which runs the proposed algorithm on the dataset and does the ablation study. It also generates a `.csv` file with the results of the algorithm which is needed as input for the `{dataset name}_visualization.ipynb` file which visualizes the results which has to be run after the dataset file.
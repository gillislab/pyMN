MetaNeighbor: a method to rapidly assess cell type identity using both functional and random gene sets
================
MetaNeighbor allows users to quantify cell type replicability across datasets using neighbor voting.

Please refer the [notebooks](./notebooks) for tutorials on how to use MetaNeighbor and consider citing [Crow et al (2018) Nature Communications](https://www.nature.com/articles/s41467-018-03282-0) if you find MetaNeighbor useful in your research.

The code was translated from R to Python, you can find the R package in this (repository)[https://github.com/gillislab/MetaNeighbor/]


## Installation

pyMN depends on mostly standard numerical and single cell RNAseq analysis python packages.
 * Numpy
 * Scipy>=1.4
 * Pandas>=.21
 * networkx
 * anndata>=.7
 * scanpy
 * matplotlib
 * seaborn
 * upsetplot

The two less standard packages are pygraphviz and bottleneck

### pygraphviz

There is currently (as of October 2020) a bug in the installation pygraphviz on pip. For mac and linux the conda installation `conda install -c anaconda pygraphviz` appears to work without issues, but for Windows you likely will need to use a distribution from this personal conda channel. `conda install -c alubbock pygraphviz`,
You can learn more about these issues [on this stackoverflow thread](https://stackoverflow.com/questions/59707234/issues-installing-pygrahviz-fatal-error-c1083-cannot-open-include-file-graph)


### bottleneck
On UNIX we haven't seen any issues with installing bottleneck using pip or conda, but on Windows we have found that, without already having Microsoft Visual C++ Build Tools already installed on your system, the easiest way to install bottleneck is with `conda install -c anaconda bottleneck`. If you have MVCBT then it should work with pip when you install pyMN. 


### Installing pyMN
You can either install this by cloning the repository, moving into it and then running:
  `python setup.py install` 

  or
   
  `pip install .` 

If you don't want to clone the repository you can just run:
	`pip install git+https://github.com/gillislab/pyMN#egg=pymn`


## Usage

For detailed tutorials refer to the [notebooks](./notebooks) folder for 3 different protocols. In general the code works similar to other packages that heavily interact with Anndata objects (like scanpy and the packages in scanpy.external). You pass an AnnData object and the parameters necesssary to run the function as you intend it to. There are 2 types of parameters.
	1) Ones that describe data already in the Anndata
	2) Ones that tune the functionality of the method


## Issues and Bugs

If you have any issues or find any bugs please use the [Issue Tracker](https://github.com/gillislab/pyMN/issues)

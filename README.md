# Probabilistic Graphical Models (PGM)

![](figures/figure_readme.png)

*Learning protein constitutive motifs from sequence data*

![](figures/figure_zebrafish.png)

*Learning neural assemblies of larval zebrafish from whole-brain neural recordings. Credit: Thijs L. van der Plas*


This repository consists of a high-level, object-oriented Python implementation of directed and undirected Probabilistic Graphical Models such as Restricted Boltzmann Machines (RBM), Boltzmann Machines (BM), Mixture of Independent (MoI), Generalized Linear Models (GLM).
PGM is implemented using numpy and numba and runs on CPU. It was originally developed for analysis of discrete multidimensional biological data such as multiple sequence alignments of proteins or neural spike recordings, see references below.

In addition to the core algorithm, the repository contains various preprocessing and visualization tools for protein sequence data; below is a short example of modeling protein sequence data:

```
sequences = Proteins_utils.load_FASTA('data/WW/WW_domain_MSA.fasta') # Load protein sequences.
sequence_weights = 1.0/Proteins_utils.count_neighbours(sequences) # Compute sequence weights.
RBM = rbm.RBM(
              visible = 'Potts', # Categorical visible data.
              hidden = 'dReLU', # double ReLU hidden units.
              n_v = sequences.shape[-1], # Data dimension (here, number of columns in MSA)
              n_cv = 21, # Number of categories (here, 20 amino acids + 1 gap)
              n_h = 50 # Number of hidden units
              ) # RBM object initialization.
RBM.fit(sequences, weights = sequence_weights, n_iter=500, N_MC=10,verbose=0, vverbose=1, l1b=0.25) # Fit by Persistent Contrastive Divergence for 500 epochs, 10 Monte Carlo steps; L_1^2 sparse regularization penalty = 0.25.

print( RBM.likelihood(sequences).mean() ) # Evaluate partition function Z by Annealed Importance Sampling, then evaluate  sequences likelihood.

artificial_sequences,artificial_sequences_representation = RBM.gen_data(Nthermalize=1000,Nchains=100,Lchains=100,Nstep=10) # Generate artificial samples by Markov Chain Monte Carlo.

position_weight_matrix = utilities.average(sequences,weights=sequence_weights,c=21) # Compute Position Weight Matrix (i.e. frequency of each amino acid at each site)
position_weight_matrix_artificial = utilities.average(artificial_sequences,c=21) # Same for artificial sequences.

sequence_logo.Sequence_logo(position_weight_matrix); # Visualize sequence logo of the MSA.
sequence_logo.Sequence_logo(position_weight_matrix_artificial); # Same, for artificial sequences.

plots_utils.make_all_weights(RBM,sequences,name = 'all_weights_WW.pdf',pdb_file='2n1o') # For each hidden unit of the trained RBM model, show weight logo, input distribution and  attached for all hidden units and map them onto a PDB structure.

```

## Requirements
- A Python3 distribution (e.g. Anaconda, https://www.anaconda.com/products/individual tested for >=3.6.8) with latest versions of numpy (>= 1.18.4) and numba (>=0.44.1). Please upgrade your packages via:
```
pip install --upgrade numpy
pip install --ignore-installed llvmlite
pip install --upgrade numba
```


- Optional dependencies, only for mapping RBM weights onto PDB structures with plots_utils.make_all_weights:
    - Biopython (For parsing PDB files)
```
pip install biopython
```
- Chimera (https://www.cgl.ucsf.edu/chimera/ for visualizing PDB structures)

- HMMer (http://hmmer.org/ , for aligning the PDB structure with the MSA)

## Installation
- No installation is required for basic usage; please refer to the notebooks in the examples/ folder.
- For mapping RBM weights on PDB structures (plots_utils.make_all_weights), please change the paths in utilities/Proteins_3D_utils.py once Chimera and HMMer are installed.

## Usage
For usage, please refer to the Jupyter notebooks in the examples/ folder for usage. Short summary of the content of each notebook:  

- **MNIST**, the handwritten digit database (See ref [1]).
  - Training RBM and BM with various hidden unit potentials
  - Artificial digit generation by Monte Carlo
  - Performance comparison.  



- **Lattice Proteins** (LP) are artificial protein sequences, used here to benchmark the algorithm against ground truth knowledge of the structure and of the fitness function (See refs [3,4]).
  - Training RBM
  - Hidden unit selection based on hidden unit properties (sparsity, importance, non-linearity...).
  - Hidden unit visualization (weight logo, input distribution)
  - Artificial sequences generation
  - Conditional and low-temperature sampling


- **WW Domain**. The WW domain is a short ($N=31$) binding domain targeting proline-rich motifs. WW domains are usually classified into 4 types, according to their binding specificity toward different motifs (e.g. PPXY, PPLP p[T/S]P...). We show here some features found by the algorithm and their connection with binding specificity. This notebook reproduces the panels of Figure 3, Ref [3].

  - Training RBM
  - Hidden unit selection based on hidden unit properties and available sequence labels.
  - Hidden unit visualization.
  - Mapping hidden units onto PDB files.
  - Artificial sequences generation.
  - Conditional and low-temperature sampling.



- **Kunitz Domain**: The Kunitz domain is a family of protease inhibitors. It is one of the simplest and shortest ($N = 53$) domain, with a well-known secondary and tertiary structure (2 beta strands, 2 alpha-helices, 3 disulfide bridges). This notebook reproduces the panels of Figure 2, Ref [3].
  - Training RBM
  - Hidden unit selection
  - Hidden unit visualization
  - Mapping hidden units onto PDB files.
  - Contact prediction


- **Hsp70 Protein**. The Hsp70 protein is a relatively large ($N \sim 650 $) chaperone protein; they can assist folding and assembly of newly synthesized proteins, trigger refolding cycles of misfolded proteins, transport unfolded proteins through organelle membranes, and when necessary, deliver non-functional proteins to the proteasome, endosome or lysosome for recycling.
Hsp70 genes differ by organism, location of expression (Nucleus/Cytoplasm, Mitochondria, ER, Chloroplasta), mode of expression (stress-induced or constitutive), substrate specificity (Target regular proteins or Iron-Sulfur cluster proteins,...). This notebook reproduces the panels of Figure 4, Ref [3].

  - Training RBM
  - Hidden unit selection based on available labels.
  - Hidden unit visualization for very long sequences.
  - Mapping hidden units onto PDB files.

- **Ising 1D**. The Ising 1D model is a statistical physics model of interacting spins. Ising 1D configurations can be generated by Monte Carlo, and RBM trained on these configurations. Harsh et al. have shown that the maximum-likelihood learning dynamics can be mapped onto a Reaction-Diffusion equation and the evolution and shape of the weights predicted analytically, see Ref. [6].
  - Generating Ising 1D configurations at various temperatures using BM.
  - Training RBM on Ising 1D samples.
  - Weight visualization.

- **XY Model** The XY model is an extension of the Ising model where spins are modeled as unit vectors instead of $\pm 1$. In Ref. [6], Harsh et al. considered a discrete variant where spin angles are discretized, and studied symmetry breaking while learning RBM.
  - Generation discrete XY model configurations.
  - Characterization of the empirical covariance matrix.

- **Curie-Weiss (Mean field Ising) Model** Same as the Ising 1D model, but in a mean-field geometry (all spins are neighbors).
  - Implementation of Curie-Weiss as a BM with uniform couplings.
  - Implementation of Curie-Weiss as a RBM with $M=1$ Gaussian hidden unit and uniform couplings.
  - Sample generation, order parameter distributions.
  - Monte Carlo trajectory.
predicted analytically

- **Equivalence between Boltzmann Machines and Gaussian RBM.** RBM with Gaussian hidden units are equivalent to BM with low-rank interaction matrices.
  - Code for mapping a Gaussian RBM onto a BM.
  - Sample generation and moments comparison.

- **Generalized Linear Model.** The core layer developed for implementing BM and RBM can be repurposed for generalized linear model. Here, we apply GLM for pseudo-likelihood inference of Ising/Potts BM couplings.
  - Ising/Potts BM with random weights
  - MC sample generation.
  - Inference of couplings by pseudo-likelihood maximization.
  - Comparison at various sample sizes.
  
- **Neural recordings of larval zebrafish spontaneous activity**
Light-sheet fluorescence microscopy is a novel imaging method for simultaneously recording the activity of whole brains of small vertebrae (larval zebrafish) at neuron scale (~40K neurons).
The measured patterns of spontaneous activity reflect a stochastic exploration of the neuronal state space that is constrained by the underlying assembly organization of neurons. 
    
This notebook contains main instructions for reproducing the results presented in Ref. [10]. We use RBM to model eight such large-scale recordings. We:
  -  Infer $\sim\!200$ neural assemblies, which are physiologically meaningful and whose various combinations form successive brain states.
  -  Accurately reproduce the mean activity and pairwise correlation statistics of the recordings despite limited number of parameters.
  -  Mathematically derive an interregional functional connectivity matrix, which is conserved across individual animals and correlates well with structural connectivity.

Data and pretrained models must be downloaded from the following repository: https://gin.g-node.org

For running this notebook, the following packages must be installed:
Seaborn; usually installed by default, otherwise:  ```pip install seaborn```
h5py; usually installed by default, otherwise: ```pip install h5py```
Ipyvolume: follow installation here: https://ipyvolume.readthedocs.io/en/latest/install.html

This notebook provides basic data visualization plots. For best visualizations, we strongly recommend using the Fishualizer package (https://bitbucket.org/benglitz/fishualizer_public/) developed by Thijs van der Plas for optimal visualization of the recording and learnt weights.
Instructions for interfacing the PGM and Fishualizer packages are provided at the end of the notebook. 



## References:

Articles using this package or a previous version.

1. Tubiana, J., & Monasson, R. (2017). Emergence of compositional representations in restricted Boltzmann machines. Physical review letters, 118(13), 138301.  
2. Cocco, S., Monasson, R., Posani, L., Rosay, S., & Tubiana, J. (2018). Statistical physics and representations in real and artificial neural networks. Physica A: Statistical Mechanics and its Applications, 504, 45-76.   
3. Tubiana, J., Cocco, S., & Monasson, R. (2019). Learning protein constitutive motifs from sequence data. Elife, 8, e39397.  
4. Tubiana, J., Cocco, S., & Monasson, R. (2019). Learning compositional representations of interacting systems with restricted boltzmann machines: Comparative study of lattice proteins. Neural computation, 31(8), 1671-1717.  
5. Rizzato, F., Coucke, A., de Leonardis, E., Barton, J. P., Tubiana, J., Monasson, R., & Cocco, S. (2020). Inference of compressed Potts graphical models. Physical Review E, 101(1), 012309. (only likelihood and entropy estimation)  
6. Harsh, M., Tubiana, J., Cocco, S., & Monasson, R. (2020). ‘Place-cell’emergence and learning of invariant data with restricted Boltzmann machines: breaking and dynamical restoration of continuous symmetries in the weight space. Journal of Physics A: Mathematical and Theoretical, 53(17), 174002.
7. Bravi, B., Tubiana, J., Cocco, S., Monasson, R., Mora, T., & Walczak, A. M. (2021). RBM-MHC: A Semi-Supervised Machine-Learning Method for Sample-Specific Prediction of Antigen Presentation by HLA-I Alleles. Cell systems, 12(2), 195-202.
8. Bravi, B., Balachandran, V. P., Greenbaum, B. D., Walczak, A. M., Mora, T., Monasson, R., & Cocco, S. (2021). Probing T-cell response by sequence-based probabilistic modeling. PLoS computational biology, 17(9), e1009297.
9. Roussel, C., Cocco, S., & Monasson, R. (2021). Barriers and dynamical paths in alternating Gibbs sampling of restricted Boltzmann machines. Physical Review E, 104(3), 034109.
10. van der Plas, T., Tubiana, J., Le Goc, G., Migault, G., Kunst, M., Baier, H., Bormuth, V., Englitz, B. & Debregeas, G. (2021) Compositional restricted boltzmann machines unveil the brain-wide organization of neural assemblies, bioRxiv
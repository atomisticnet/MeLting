# MeLting
Machine-learning models for melting-temperature prediction

<p align="center">
<img width="450" src="./melting_figure.svg" />
</p>
    
Common use cases for the tools:

* Materials feature construction;
* Melting temperature prediction of binary ionic materials using:
    * direct supervised learning approach, and
    * combination of supervised and unsupervised learning approaches.

## References

If you make use of this code, please cite the MeLting reference publication

Vahe Gharakhanyan, Luke J. Wirth, Jose A. Garrido Torres, Ethan Eisenberg, Ting Wang, Dallas R. Trinkle, Snigdhansu Chatterjee, Alexander Urban; Discovering melting temperature prediction models of inorganic solids by combining supervised and unsupervised learning. J. Chem. Phys. 28 May 2024; 160 (20): 204112. https://doi.org/10.1063/5.0207033

## Installation

Installation with `pip`:

```
pip install --user .
```

Or in editable (developer) mode:

```
pip install --user -e .
```

## Usage

See the [tutorials](./tutorials) subdirectory for Jupyter notebooks that demonstrate the usage of the package. 

Data files include melting temperature values, Materials Project identifiers (mp-id's) of the selected structures and all materials features used. Compound features are obtained directly from the Materials Project, when available, or through our own DFT calculations.

The presented work is applicable to binary materials only but simple engineering should lead these models to be applied to more complex (and simpler) compositions as well.


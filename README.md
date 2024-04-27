# MeLting
Machine-learning models for melting-temperature prediction

<p align="center">
<img width="450" src="./melting_figure.svg" />
</p>
    
Common use cases for the tools

* Materials feature construction;
* Melting temperature prediction of binary ionic materials using a direct supervised learning approach;
* Melting temperature prediction of binary ionic materials using a combination of supervised and unsupervised learning approaches.

## Usage

Data files include melting temperature values, Materials Project identifiers (mp-id's) of the selected structures and all materials features used. Compound features are obtained directly from the Materials Project, when available, or through our own DFT calculations.

The presented work is applicable to binary materials only but simple engineering should lead these models to be applied to more complex (and simpler) compositions as well.

## References

If you make use of this code, please cite the MeLting reference publication

[1] V. Gharakhanyan, L. J. Wirth, J. A. Garrido Torres, E. Eisenberg, T. Wang, D. R. Trinkle, S. Chatterjee, and A. Urban, *arxiv* (2024), https://arxiv.org/pdf/2403.03092

# BNL-SULI-Program-2026
My final project for Brookhaven National Laboratory's Student Undergraduate Laboratory Internship (SULI) Program for Summer 2026. A program, and some supplementary code, for lightweight PXRD analysis.

A quick tour of the repository:
- _simulated_correlation_prototype.ipynb_ is the main program for the PXRD analysis. The bottommost cell is the one with the actual function call, so just change up the parameters you need to and you're good to go. You can modify some important parameters (namely wavelength) in earlier cells, but all are labeled with a header comment, so you should be able to figure out which one's which. Additionally, there is a function for determining optimal lattice parameters for a candidate phase given the phase and an experimental data, but this is not called in the main function. You can still try it out if you want!
- _materials_project_query.py_ was the original code for requesting CIF files from the Materials Science API and saving them to the computer. Its modified version is in a cell in _simulated_correlation_prototype.ipynb_, but the original was kept just in case.
- _simulation_xrd.py_ was the original code for a lot of the simulation of CIF patterns that is used now for correlation comparisons. Its modified version is in a cell in _simulated_correlation_prototype.ipynb_, but the original was kept just in case.

Credit to Dr. Hui Zhong (National Synchrotron Light Source II (NSLS-II) Department, Brookhaven National Laboratory, Upton, NY, 11973) for the original code for _materials_project_query.py_ and _simulation_xrd.py_.

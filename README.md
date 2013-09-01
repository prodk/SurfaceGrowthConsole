SurfaceGrowthConsole

SurfaceGrowthConsole - a CUDA-based code for classical molecular dynamics simulations.
__________________________________________________________________
The current Release is for Windows. However, the code can be compiled on Linux using the Makefile. Tested on Windows XP, Windows 7 Ultimate x64, Fedora Linux with CUDA 3.2 or 5.0 and GPUs GeForce GTX 260, GeForce GT 460, GeForce GTX 480, GeForce GT 560.

How to use: Run the executable in the Release directory from the command line.
_______________________________________________________________________
Input file should have the name 'sugr_in.txt' and be in the same directory as the executable. 

Output is also saved in that directory: 
1) Diffuse.txt - time dependencies of the diffusion constant.
2) sg_...txt - measurements of different quantities.
3) rdf_...txt - radial distribution function.
4) sugr_...pdb - data for visualization in vmd.

___________________________________________________
'doc' directory contains the documentation in English and Russian. The user's guide is for the older version of the SurfaceGrowth that contained GUI and used a cutil library. The current version neither contains GUI nor uses cutil, and the input is taken from the input file. However, most of the input fields in the input file correspond to the input fields from the GUI in the guide.

Note that the manuscript 'prodanov_cpc_2010.pdf' describing the algorithms and the code was written several years ago, so the quality of English is not perfect, but is enough for understanding.

___________________________________________________
(c) Mykola Prodanov, 2010-2013, Sumy, Ukraine, Juelich, Germany. 
This code is free to use and/or distribute. However, the code is provided 'as is', without any warranty that it doesn't contain bugs and will not lead to harmful effects to the software (e.g. OS) and/or hardware where the program is invoked. The author is not liable for such harmful effects if they happen as a result of using/running the code/program.
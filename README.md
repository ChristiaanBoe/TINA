# TINA

TINA is a framework that allows representing non-NN dataflow algorithms as a series of convolutional and fully connected NN layers. This makes it possible to execute non-NN algorithms on NN HW accelerators, as well as ensure the portability of TINA implementations to any platform that supports such NN HW.

----------------------------------------------------------------------------------
# Requirements
The basic requirements of TINA are:
Numpy, Pytorch, ONNX

From that point onwards you can accelerate using whatever NN accelerator you prefer!

----------------------------------------------------------------------------------
# Usage of TINA
After the installation of the aforementioned required libraries, the TINA layers can be imported via the directory TINA layers and used in a similar manner as Pytorch.nn layers.

Example notebooks of TINA layers accelerated using the AMD Ryzen 9 7940HS can be found in the directory NPU scripts

----------------------------------------------------------------------------------
# Contact
You can contact me using the following email: c.boerkamp@tudelft.nl

----------------------------------------------------------------------------------
# Publications
If you use this work, please cite the following publication. 
Christiaan Boerkamp, Steven van der Vlugt and Zaid Al-Ars, "TINA: Acceleration of Non-NN Signal Processing Algorithms Using NN Accelerators", IEEE Int'l Workshop on Machine Learning for Signal Processing (MLSP), 2024 (https://arxiv.org/abs/2408.16551v1)

----------------------------------------------------------------------------------
# Applications
The TINA framewok has been applied to a Ryzen 9 NPU in the [Pervasive AI Developer Contest](https://www.hackster.io/contests/amd2023).
In the project [TINA: Running non NN algorithms on an AMD Ryzen NPU!](https://www.hackster.io/tina/tina-running-non-nn-algorithmns-on-an-amd-ryzen-npu-0cc58c).
Refer to this project page for an elaborate explanation.

----------------------------------------------------------------------------------
# license
Apache 2.0

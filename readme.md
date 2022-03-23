test:
python test.py

checkpoint\epoch_X4.pth: the weight of X4 SR. The more checkpoint will be available soon.

model\PRNet_ACOS.py : the CPSRNet network(ours). 

image\ : partial testing set.

partial test set.:(images from Miki[1], Simeng[2] and DoFP30([3]+Polar-SR))

[1]Morimatsu M, Monno Y, Tanaka M, et al. Monochrome and color polarization demosaicking using edge-aware 
residual interpolation[C]//2020 IEEE International Conference on Image Processing (ICIP). IEEE, 2020: 2571-2575.

[2]Qiu S, Fu Q, Wang C, et al. Linear Polarization Demosaicking for Monochrome and Colour Polarization 
Focal Plane Arrays[C]//Computer Graphics Forum. 2021.

[3]Li R, Qiu S, Zang G, et al. Reflection separation via multi-bounce polarization state 
tracing[C]//European Conference on Computer Vision. Springer, Cham, 2020: 781-796.

Note: The training dataset and testing datasets are too huge to upload as supplementary material.
The novel training dataset polar-SR and testing dataset DoFP30  introduced in this paper will be made 
publicly available with a license that allows free usage for research purposes.


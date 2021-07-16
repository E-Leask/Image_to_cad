import numpy as np
from sympy import *
ab_line=np.array([100000,-23333],dtype=np.int_)
cd_line=np.array([100000,-233331],dtype=np.int_)
x, y= symbols('x y')
int_point,=linsolve([-y + ab_line[0]*x + ab_line[1],-y + cd_line[0]*x + cd_line[1]],(x,y))
print(int_point)
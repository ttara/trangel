#This program calculates the orthocentre of Triangle ABC
import numpy as np
from coeffs import *

A = np.array([-2,-2]) 
B = np.array([1,3]) 
C = np.array([4,-1]) 

p = np.zeros(2)

#AP
n1 = dir_vec(B,C)
p[0] = n1@A
#BQ
n2 = dir_vec(C,A)
p[1] = n2@B

#Intersection
N=np.vstack((n1,n2))
H=np.linalg.inv(N)@p
print(H)




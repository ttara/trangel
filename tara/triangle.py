import numpy as np
import matplotlib.pyplot as plt


A = np.array([-2,-2])
B = np.array([1,3])
C = np.array([4,-1])
len=10
x_AB = np.zeros((2,len))
lam1 = np.linspace(0,1,len)
for i in range(len):
	temp1 = A+lam1[i]*(B-A)
	x_AB[:,i] = temp1.T

x_BC = np.zeros((2,len))
lam2 = np.linspace(0,1,len)
for i in range(len):
	temp2 = B+lam2[i]*(C-B)
	x_BC[:,i] = temp2.T

x_CA = np.zeros((2,len))
lam3 = np.linspace(0,1,len)
for i in range(len):
	temp3 = C+lam3[i]*(A-C)
	x_CA[:,i] = temp3.T

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.grid()	
plt.plot(A[0],A[1],'o')
plt.text(B[0]*(1-0.2),B[1]*(1), 'B')

plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.grid()	
plt.plot(B[0],B[1],'o')
plt.text(C[0]*(1-0.2),C[1]*(1), 'C')

plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.grid()	
plt.plot(C[0],C[1],'o')
plt.text(A[0]*(1-0.2),A[1]*(1), 'A')

plt.show()

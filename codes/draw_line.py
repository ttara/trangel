#Plotting AB
import numpy as np
import matplotlib.pyplot as plt

#if using termux
import subprocess
import shlex
#end if


A = np.array([-2,-2]) 
B = np.array([1,3]) 
len =10

x_AB = np.zeros((2,len))
lam = np.linspace(0,1,len)
for i in range(len):
  temp1 = A + lam[i]*(B-A)
  x_AB[:,i]= temp1.T

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.grid() # minor

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')

#if using termux
plt.savefig('../figs/draw_line.pdf')
plt.savefig('../figs/draw_line.eps')
subprocess.run(shlex.split("termux-open ../figs/draw_line.pdf"))
#else
#plt.show()




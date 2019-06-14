import numpy as np
import matplotlib.pyplot as plt


A = np.array([-2,-2])
B = np.array([1,3])
C = np.array([4,-1])
D = (A+B)/2
E = (B+C)/2
F = (C+A)/2


def line(A,B):
	len=10
	x_AB = np.zeros((2,len))
	lam1 = np.linspace(0,1,len)
	for i in range(len):
		temp1 = A+lam1[i]*(B-A)
		x_AB[:,i] = temp1.T
	return x_AB
lineAB=line(A,B)
lineBC=line(B,C)
lineCA=line(C,A)
lineCD=line(C,D)
lineAE=line(A,E)
lineBF=line(B,F)

n_BF=np.matmul(np.array([[0,1],[-1,0]]),F-B)
n_AE=np.matmul(np.array([[0,1],[-1,0]]),E-A)

p=np.zeros((2,1))
p[0]=np.matmul(n_BF,B)
p[1]=np.matmul(n_AE,A)
 
N=np.vstack((n_BF,n_AE))
 
x=np.matmul(np.linalg.inv(N),p)
print(x)

plt.plot(lineAB[0,:],lineAB[1,:],label='$AB$')
plt.grid()	
plt.plot(A[0],A[1],'o')
plt.plot(x[0],x[1],'o')

plt.text(B[0]*(1-0.2),B[1]*(1), 'B')

plt.plot(lineBC[0,:],lineBC[1,:],label='$BC$')
plt.grid()	
plt.plot(B[0],B[1],'o')
plt.text(C[0]*(1-0.2),C[1]*(1), 'C')

plt.plot(lineCA[0,:],lineCA[1,:],label='$CA$')
plt.grid()	
plt.plot(C[0],C[1],'o')
plt.text(A[0]*(1-0.2),A[1]*(1), 'A')

plt.plot(lineCD[0,:],lineCD[1,:],label='$CD$')
plt.grid()	
plt.plot(D[0],D[1],'o')
plt.text(D[0]*(1-0.2),D[1]*(1), 'D')

plt.plot(lineAE[0,:],lineAE[1,:],label='$AE$')
plt.grid()	
plt.plot(E[0],E[1],'o')
plt.text(E[0],E[1], 'E')

plt.plot(lineBF[0,:],lineBF[1,:],label='$BF$')
plt.grid()	
plt.plot(F[0],F[1],'o')
plt.text(F[0]*(1-0.2),F[1]*(1), 'F')

plt.show()

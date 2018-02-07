import sys
import numpy as np
import numpy.linalg as la

# Assignment1: Gradient Descent Optimization
# Valdez, Paolo F. | MSEE Student No. 2009-11407

# Assumptions:
# for third degree equations: C3 C2 C1 C0
# for second degree equations:   C2 C1 C0
# for first degree equations:       C1 C0

#Answer:
#For the given input: 2 4 -1 The recommended learning rate is 𝝐 = 1.5625e-05 given tolerance value, 𝛿 = 0.01

tolerance = 0.01

# Input fxn
# Splits spaces on integer and places in corresponding degree here
# Resizes parameter matrix into column vector
numbers = list(map(float,input().split()))
numlen = len(numbers)
numbers=np.asarray(numbers, dtype=float).reshape(-1,1)


#good for 3rd degree, based on trial and error experimentation
if numlen == 4:
    learn_rate = 0.0001

#good for 2nd degree function, based on trial and error experimentation
if numlen == 3:
    learn_rate = 0.001
#good for 1st degree function, based on trial and error experimentation
if numlen == 2:
    learn_rate = 0.001

if numlen == 1:
    learn_rate = 0.001

coefficient_mat = np.array(numbers, dtype=float)
coefficient_mat = coefficient_mat.reshape(-1,1)
#print("original weight matrix: ")
#print(coefficient_mat)
#print(" ")

    #   1st degree  A
input_mat1 = np.array([],dtype=float)
input_mat1 = np.arange(-10,10,0.5,dtype=float) # generates a random matrix for the input matrix A
input_mat1 = input_mat1.reshape(-1,1)
dims = input_mat1.size
    #   2nd degree A^2
input_mat2 = np.array([], dtype=float)
input_mat2 = np.power(input_mat1,2)
input_mat2 = input_mat2.reshape(-1,1)
    #   3rd degree A^3
input_mat3 = np.array([], dtype=float)
input_mat3 = np.power(input_mat1,3)
input_mat3 = input_mat3.reshape(-1,1)
#print ('m3',input_mat3)
    #    0th degree
input_mat0 = np.array([], dtype=float)
input_mat0 = np.ones(dims)
input_mat0 = input_mat0.reshape(-1,1)

input_mat=np.array([])

if numlen == 4:
    input_mat=np.concatenate((input_mat3, input_mat2, input_mat1, input_mat0),axis=1)
if numlen == 3:
    input_mat=np.concatenate((input_mat2, input_mat1, input_mat0),axis=1)
if numlen == 2:
    input_mat=np.concatenate((input_mat1, input_mat0),axis=1)
if numlen == 1:
    input_mat=np.concatenate((input_mat0))

#  Ax = b , Solve for output matrix, b
output_mat = np.matmul(input_mat,coefficient_mat)
# adds uniform noise to ouput matrix, b
output_mat_noise = output_mat + np.random.uniform(-1.0,1.0,output_mat.shape)

# generates a new coefficient matrix, x'
new_coefficient_mat = np.random.uniform(0.0,1.0,numlen)
new_coefficient_mat = new_coefficient_mat.reshape([-1,1])

#   solves for ' A^T A'
input_mat_T = np.transpose(input_mat)
#print("A^T")
#print(input_mat_T)
loss_fxn1 = np.matmul(input_mat_T, input_mat)
#print("A^T A :")
#print(loss_fxn1)
#print(" ")

#  solves for 'A^T A x'
loss_fxn2 = np.matmul(loss_fxn1, new_coefficient_mat)
print("A^T A x")
print(loss_fxn2)
print(" ")
#  solves for 'A^T b'
loss_fxn3 = np.matmul(input_mat_T,output_mat_noise)
print("A^T b")
print(loss_fxn3)
print(" ")
#   solves for ' A^T Ax - A^T b '
loss_fxn0 = loss_fxn2 - loss_fxn3
print("A^T A x - A^T b")
print(loss_fxn0)
print(" ")
#   solves for ||A^T A x - A^T b||2
loss_fxn_l2 = la.norm(loss_fxn0, ord = 'fro')
print("||n|2")
print(loss_fxn_l2)
print(" ")

iter = 0
while loss_fxn_l2 > tolerance:

    old_loss_fxn_l2 = loss_fxn_l2
    # x(1) ← x(0) - 𝝐 ( A^T A x(0) - A^T b)
    new_coefficient_mat = new_coefficient_mat - (learn_rate*loss_fxn0)

    #solves for new L2 norm based on x(1): ||A^T A x(1) - A^T b||2
    loss_fxn2 = np.matmul(loss_fxn1, new_coefficient_mat)
    loss_fxn0 = loss_fxn2 - loss_fxn3
    loss_fxn_l2 = la.norm(loss_fxn0, ord = 'fro')

    #decreases learning rate to avoid 'overshooting' if current loss function value greater than previous loss function value
    if loss_fxn_l2 > old_loss_fxn_l2:
        learn_rate = 0.5*learn_rate

    #if there is no change in the loss function, program stops
    if old_loss_fxn_l2 == loss_fxn_l2:
        sys.exit()

    iter +=1
    print(new_coefficient_mat, 'loss fxn value: ', loss_fxn_l2, 'learn rate:', learn_rate, 'iteration:', iter)

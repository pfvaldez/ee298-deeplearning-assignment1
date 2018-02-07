#Assumptions:
#for third degree equations: C3 C2 C1 C0
#for second degree equations:   C2 C1 C0
#for first degree equations:       C1 C0

import sys
import numpy as np
import numpy.linalg as la

#! Splits spaces on integer and places in correspondging degree here
#..Resizes parameter matrix into n x 1 matrix
tolerance = 0.1

#Input fxn
numbers = list(map(float,input().split()))
numlen = len(numbers)
numbers=np.asarray(numbers, dtype=float).reshape(-1,1)

#good for 3rd degree
if numlen == 4:
    learn_rate = 0.0001

#good for 2nd degree function
if numlen == 3:
    learn_rate = 0.001

if numlen == 2:
    learn_rate = 0.001

if numlen == 1:
    learn_rate = 0.001

coefficient_mat = np.array(numbers, dtype=float)
coefficient_mat = coefficient_mat.reshape(-1,1)
#print("original weight matrix: ")
#print(coefficient_mat)
#print(" ")

    #   1st degree
input_mat1 = np.array([],dtype=float)
#good for 2nd degree function
input_mat1 = np.arange(-10,10,0.5,dtype=float)
input_mat1 = input_mat1.reshape(-1,1)
dims = input_mat1.size
    #   second degree
input_mat2 = np.array([], dtype=float)
input_mat2 = np.power(input_mat1,2)
input_mat2 = input_mat2.reshape(-1,1)
    #   third degree
input_mat3 = np.array([], dtype=float)
input_mat3 = np.power(input_mat1,3)
input_mat3 = input_mat3.reshape(-1,1)
#print ('m3',input_mat3)
    #    0th degree
input_mat0 = np.array([], dtype=float)
input_mat0 = np.ones(dims)
input_mat0 = input_mat0.reshape(-1,1)

input_mat=np.array([])
#print("x3",input_mat3)
#print("x1",input_mat1)
#print("x0",input_mat0)
if numlen == 4:
    input_mat=np.concatenate((input_mat3, input_mat2, input_mat1, input_mat0),axis=1)
if numlen == 3:
    input_mat=np.concatenate((input_mat2, input_mat1, input_mat0),axis=1)
if numlen == 2:
    input_mat=np.concatenate((input_mat1, input_mat0),axis=1)
if numlen == 1:
    input_mat=np.concatenate((input_mat0))
# 'A'
print("A, input matrix")
print(input_mat)
print(" ")
# 'b'
output_mat = np.matmul(input_mat,coefficient_mat)
print("b, output matrix:")
print(output_mat)
print(" ")
#!!!!!!!!!!!!!!!problems here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11111111111111111111
output_mat_noise = output_mat + np.random.uniform(-1.0,1.0,output_mat.shape)
print("output matrix with noise:")
print(output_mat)
print(" ")
# 'x'

new_coefficient_mat = np.random.uniform(0.0,1.0,numlen)
new_coefficient_mat = new_coefficient_mat.reshape([-1,1])
print("new_coefficient_mat")
print(new_coefficient_mat)
print(" ")

#   'A^T A'
input_mat_T = np.transpose(input_mat)
print("A^T")
print(input_mat_T)
loss_fxn1 = np.matmul(input_mat_T, input_mat)
print("A^T A :")
print(loss_fxn1)
print(" ")

#   'A^T A x'
loss_fxn2 = np.matmul(loss_fxn1, new_coefficient_mat)
print("A^T A x")
print(loss_fxn2)
print(" ")
#   'A^T b'
loss_fxn3 = np.matmul(input_mat_T,output_mat_noise)
print("A^T b")
print(loss_fxn3)
print(" ")
#   A^T Ax - A^T b
loss_fxn0 = loss_fxn2 - loss_fxn3
print("A^T A x - A^T b")
print(loss_fxn0)
print(" ")
#   ||A^T A x - A^T b||2
loss_fxn_l2 = la.norm(loss_fxn0, ord = 'fro')
print("||A^T A x - A^T b||2")
print(loss_fxn_l2)
print(" ")
iter = 0
while loss_fxn_l2 > tolerance:
    old_loss_fxn_l2 = loss_fxn_l2
    new_coefficient_mat = new_coefficient_mat - (learn_rate*loss_fxn0)
    loss_fxn2 = np.matmul(loss_fxn1, new_coefficient_mat)
    loss_fxn0 = loss_fxn2 - loss_fxn3
    loss_fxn_l2 = la.norm(loss_fxn0, ord = 'fro')
    if loss_fxn_l2 > old_loss_fxn_l2:
        learn_rate = 0.5*learn_rate
    #if loss_fxn_l2 < old_loss_fxn_l2:
    #    learn_rate = 1.1*learn_rate
    if old_loss_fxn_l2 == loss_fxn_l2:
        sys.exit()
    iter +=1
    print(new_coefficient_mat, 'loss fxn value: ', loss_fxn_l2, 'learn rate:', learn_rate, 'iteration:', iter)

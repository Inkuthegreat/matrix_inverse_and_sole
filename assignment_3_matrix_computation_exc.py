#!/usr/bin/env python
# coding: utf-8
"""
-----------------------------------------------------------------------------
Assignment-3: Matrix computation exercise using python and numpy
-----------------------------------------------------------------------------
AUTHOR: Soumitra Samanta (soumitra.samanta@gm.rkmvu.ac.in)
-----------------------------------------------------------------------------
Package required:
Numpy: https://numpy.org/
Matplotlib: https://matplotlib.org
-----------------------------------------------------------------------------
"""

import numpy as np
from numpy.random import RandomState
import time
import math as m
from typing import Tuple

__all__ = [
    'rand_vec',
    'vector_dot_product',
    'rand_matrix',
    'matrix_multiplication',
    'elementary_ops_interchange_rows',
    'elementary_ops_scale_row',
    'elementary_ops_change_row',
    'elementary_ops_interchange_rows_mult',
    'elementary_ops_scale_row_mult',
    'elementary_ops_change_row_mult',
    'elementary_ops_interchange_cols',
    'elementary_ops_scale_col',
    'elementary_ops_change_col',
    'elementary_ops_interchange_cols_mult',
    'elementary_ops_scale_col_mult',
    'elementary_ops_change_col_mult',
    'sweep_out_row',
    'sweep_out_row_mult',
    'sweep_out_col',
    'sweep_out_col_mult',
    'reduced_echelon_form',
    'reduced_upper_triangular_form',
    'reduce_non_sigular_matrix_to_identity',
    'generate_random_linear_system',
    'soln_n_equations_n_unknowns',
    
]

def random_no_gen(n):
    v=[]
    seed=time.time()
    modulus = 2**19
    multiplier = 1203512201
    increment = 17749
    for i in range(0,n):
        seed=(seed*multiplier+increment) % modulus
        v.append(seed/modulus)
        
    return v

def rand_vec(n,l):
    v=[]
    rand_float=[]
    span=2*l+1
    if(l>0):
        rand_float=random_no_gen(n)
        for i in range(0,n):
            v.append(int(span*(rand_float[i]))-l)
    else:
        print("ValueError:{l} must be positive")

    return v



def vector_dot_product(
    x: np.array,
    y: np.array,
)->float:
    """
    Dot product between two vectors
    
    Inputs:
        - x: 1st vector
        - y: 2nd vector
       
    Output:
        - val: Dot product between x and y
    
    """
    
    val = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    val=0
    if (len(x)==len(y)):
        for i in range(0,len(x)):
            val+=(x[i]*y[i])
    else:
        print("ValueError: length didnot match")
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
    
    return val

def rand_matrix(m,n,l):
    v=[]
    for _ in range(0,m):
        v.append(rand_vec(n,l))
    return v





def matrix_multiplication(
    x: np.array,
    y: np.array,
)->np.array:
    """
    Dot product between two vectors
    
    Inputs:
        - x: 1st matrix
        - y: 2nd matrix
       
    Output:
        - val: multiplication x and y
    
    """
    
    val = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    m=len(x)
    p_1=len(x[0])
    p_2=len(y)
    n=len(y[0])
    val=[[0 for _ in range(n)] for _ in range(m) ]
    if(p_1 == p_2):
        for i in range(m):
            for j in range(n):
                for k in range(p_1):
                    val[i][j]+=float(x[i][k])*float(y[k][j])
    else:
        print("Matrix multiplication doesnot possible")
            
    
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
    
    return val


def elementary_ops_interchange_rows(
    A: np.array, 
    ith_row: int, 
    jth_row: int
)->np.array:
    """
    Elementary row operation: interchange i-th and j-th rows
    
     Inputs:
        - A: Given matrix
        - ith_row: i-th row
        - jth_row: j-th row
        
    Output:
        - A: Resultant rows interchhanged matrix
    """
    
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    if not (0<ith_row<=len(A) and 0<jth_row<=len(A) and ith_row!=jth_row):
        return "OOB error or did not swap possible"
    A[ith_row-1],A[jth_row-1]=A[jth_row-1],A[ith_row-1]
    
    
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A


def elementary_ops_scale_row(
    A: np.array, 
    ith_row: int, 
    scalar_val: float
)->np.array:
    """
    Elementary row operation: Scaling i-th row  with the scalar_val value
    
     Inputs:
        - A: Given matrix
        - ith_row: row want to scale
        - scalar_val: scalar value
        
    Output:
        - A: Resultant row scaling matrix
    """
    
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    if not (0<ith_row<=len(A)):
        return "OOB error"
    A[ith_row-1]=[scalar_val * float(element) for element in A[ith_row-1]]
            
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A


def elementary_ops_change_row(
    A: np.array, 
    ith_row: int, 
    jth_row: int,
    scalar_val: float
)->np.array:
    """
    Elementary row operation: Change i-th based on i<-i + scalar_val*j
    
     Inputs:
        - A: Given matrix
        - ith_row: i-th row
        - jth_row: j-th row
        - scalar_val: scalar value
        
    Output:
        - A: Resultant row updated matrix
    """
    
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    if not (0<ith_row<=len(A) and 0<jth_row<=len(A) and ith_row!=jth_row):
        return "OOB error or did not operation possible"
    A[ith_row-1]=[float(element_i)+scalar_val*float(element_j) for (element_i,element_j) in zip(A[ith_row-1],A[jth_row-1])]
            
            
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A


def elementary_ops_interchange_rows_mult(
    A: np.array, 
    ith_row: int, 
    jth_row: int
)->Tuple[np.array, np.array]:
    """
    Elementary row operation: interchange i-th and j-th rows
    
     Inputs:
        - A: Given matrix
        - ith_row: i-th row
        - jth_row: j-th row
        
    Output:
        - A: Resultant rows interchhanged matrix
    """
    
    I = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    I=[[(1 if k==m else 0)for k in range(len(A))]for m in range(len(A))]
    I=elementary_ops_interchange_rows(I,ith_row,jth_row)
    #print(I)
    A=matrix_multiplication(I,A)
    
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A, I


def elementary_ops_scale_row_mult(
    A: np.array, 
    ith_row: int, 
    scalar_val: float
)->Tuple[np.array, np.array]:
    """
    Elementary row operation: Scaling i-th row  with the scalar_val value
    
     Inputs:
        - A: Given matrix
        - ith_row: row want to scale
        - scalar_val: scalar value
        
    Output:
        - A: Resultant row scaling matrix
    """
    
    I = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    I=[[(1 if k==m else 0)for k in range(len(A))]for m in range(len(A))]
    I=elementary_ops_scale_row(I,ith_row,scalar_val)
    #print(I)
    A=matrix_multiplication(I,A)
    
            
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A, I


def elementary_ops_change_row_mult(
    A: np.array, 
    ith_row: int, 
    jth_row: int,
    scalar_val: float
)->Tuple[np.array, np.array]:
    """
    Elementary row operation: Change i-th based on i<-i + scalar_val*j
    
     Inputs:
        - A: Given matrix
        - ith_row: i-th row
        - jth_row: j-th row
        - scalar_val: scalar value
        
    Output:
        - A: Resultant row updated matrix
    """
    
    I = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    I=[[(1 if k==m else 0)for k in range(len(A))]for m in range(len(A))]
    I=elementary_ops_change_row(I,ith_row,jth_row,scalar_val)
    #print(I)
    A=matrix_multiplication(I,A)
   
    
            
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A, I


def elementary_ops_interchange_cols(
    A: np.array, 
    ith_col: int, 
    jth_col: int
)->np.array:
    """
    Elementary col operation: interchange i-th and j-th cols
    
     Inputs:
        - A: Given matrix
        - ith_col: i-th col
        - jth_col: j-th col
        
    Output:
        - A: Resultant cols interchhanged matrix
    """
    
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#

    if not (0<ith_col<=len(A) and 0<jth_col<=len(A) and ith_col!=jth_col):
        return "OOB error or did not swap possible"
    for k in range(len(A)):
        A[k][ith_col-1],A[k][jth_col-1]=A[k][jth_col-1],A[k][ith_col-1]
    
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A


def elementary_ops_scale_col(
    A: np.array, 
    ith_col: int, 
    scalar_val: float
)->np.array:
    """
    Elementary col operation: Scaling i-th col  with the scalar_val value
    
     Inputs:
        - A: Given matrix
        - ith_col: col want to scale
        - scalar_val: scalar value
        
    Output:
        - A: Resultant col scaling matrix
    """
    
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    if not (0 <ith_col<=len(A)):
        return "OOB error "
    for k in range(len(A)):
        A[k][ith_col-1]=scalar_val* float(A[k][ith_col-1])
         
            
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A


def elementary_ops_change_col(
    A: np.array, 
    ith_col: int, 
    jth_col: int,
    scalar_val: float
)->np.array:
    """
    Elementary col operation: Change i-th col based on i<-i + scalar_val*j
    
     Inputs:
        - A: Given matrix
        - ith_col: i-th col
        - jth_col: j-th col
        - scalar_val: scalar value
        
    Output:
        - A: Resultant col updated matrix
    """
    
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    if not (0<ith_col<=len(A[0]) and 0<jth_col<=len(A[0]) and ith_col!=jth_col):
        return "OOB error or did not operation possible"
    for k in range(len(A)):
        A[k][ith_col-1]=((float(A[k][jth_col-1])*scalar_val) + float(A[k][ith_col-1]))
      
            
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A


def elementary_ops_interchange_cols_mult(
    A: np.array, 
    ith_col: int, 
    jth_col: int
)->np.array:
    """
    Elementary col operation: interchange i-th and j-th cols
    
     Inputs:
        - A: Given matrix
        - ith_col: i-th col
        - jth_col: j-th col
        
    Output:
        - A: Resultant cols interchhanged matrix
    """
    
    I = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    I=[[(1 if k==m else 0)for k in range(len(A[0]))]for m in range(len(A[0]))]
    I=elementary_ops_interchange_rows(I,ith_col,jth_col)
    A=np.array(matrix_multiplication(A,I))
    
    
    
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A, I


def elementary_ops_scale_col_mult(
    A: np.array, 
    ith_col: int, 
    scalar_val: float
)->Tuple[np.array, np.array]:
    """
    Elementary col operation: Scaling i-th col  with the scalar_val value
    
     Inputs:
        - A: Given matrix
        - ith_col: col want to scale
        - scalar_val: scalar value
        
    Output:
        - A: Resultant col scaling matrix
    """
    
    I = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    I=[[(1 if k==m else 0)for k in range(len(A[0]))]for m in range(len(A[0]))]
    I=elementary_ops_scale_row(I,ith_col,scalar_val)
    A=np.array(matrix_multiplication(A,I))
    
            
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A, I


def elementary_ops_change_col_mult(
    A: np.array, 
    ith_col: int, 
    jth_col: int,
    scalar_val: float
)->Tuple[np.array, np.array]:
    """
    Elementary col operation: Change i-th col based on i<-i + scalar_val*j
    
     Inputs:
        - A: Given matrix
        - ith_col: i-th col
        - jth_col: j-th col
        - scalar_val: scalar value
        
    Output:
        - A: Resultant col updated matrix
    """
    
    I = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    I=[[(1 if k==m else 0)for k in range(len(A[0]))]for m in range(len(A[0]))]
    I=elementary_ops_change_row(I,jth_col,ith_col,scalar_val)
    A=np.array(matrix_multiplication(A,I))
    
            
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A, I


def sweep_out_row(
    A: np.array,
    ith_row: int,
    pivot_element: Tuple[int, int]
)->np.array:
    """
    Sweep out a i-th row based on the given pivot element
    
    Inputs:
        - A: Given matrix
        - ith_row: row to sweep out
        - pivot_element pivot element
        
    Output:
        - A: Resultant row sweep out matrix
    """
    
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    pivot_row,pivot_col=pivot_element
    if pivot_row < 1 or pivot_row > len(A) or pivot_col < 1 or pivot_col > len(A[0]):
        return "Pivot indices out of bounds"
    pivot_val=float(A[pivot_row-1][pivot_col-1])
    #print("Pivot val= ", pivot_val)
    if pivot_val==0:
        return "Error Pivotal Value must be non-zero"
    scaling=float(1/pivot_val)
    elementary_ops_scale_col(A,pivot_col,scaling)
    #print(A)
    for temp_j in range(len(A[0])):
        if(temp_j!=pivot_col-1):
            multiplier=-float(A[pivot_row-1][temp_j])
            #print(multiplier)
            elementary_ops_change_col(A,temp_j+1,pivot_col,multiplier)
            #print(A)
            #print(temp_j)
    
    
        
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
    
    return A

    
def sweep_out_row_mult(
    A: np.array,
    ith_row: int,
    pivot_element: Tuple[int, int]
)->np.array:
    """
    Sweep out a i-th row based on the given pivot element
    
    Inputs:
        - A: Given matrix
        - ith_row: row to sweep out
        - pivot_element pivot element
        
    Output:
        - A: Resultant row sweep out matrix
    """
    
    E_ops = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    row,col = pivot_element
    m = len(A)
    n = len(A[0])
    E_ops=[[(1 if k==m_1 else 0)for k in range(n)]for m_1 in range(n)]
    scaling=1/float(A[row-1][col-1])
    elementary_ops_scale_col(E_ops,col,scaling)
    for temp in range(n):
        if temp!=col-1:
            I=[[(1 if k==m_2 else 0)for k in range(len(A[0]))]for m_2 in range(len(A[0]))]
            multiplier=-float(A[row-1][temp])
            elementary_ops_change_col(E_ops,temp+1,col,multiplier)
            E_ops = matrix_multiplication(E_ops,I)
    A=matrix_multiplication(A,E_ops)
    
        
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
    
    return A, E_ops
    
    
def sweep_out_col(
    A: np.array,
    ith_col: int,
    pivot_element: Tuple[int, int]
)->np.array:
    """
    Sweep out a i-th column based on the given pivot element
    
    Inputs:
        - A: Given matrix
        - ith_col: column to sweep out
        - pivot_element pivot element
        
    Output:
        - A: Resultant column sweep out matrix
    """
    
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    pivot_row,pivot_col=pivot_element
    if pivot_row < 1 or pivot_row > len(A) or pivot_col < 1 or pivot_col > len(A[0]):
        return "Pivot indices out of bounds"
    pivot_val=float(A[pivot_row-1][pivot_col-1])
    if pivot_val==0:
        return "Error Pivotal Value must be non-zero"
    scaling=float(1/pivot_val)
    elementary_ops_scale_row(A,pivot_row,scaling)
    for temp_i in range(len(A)):
        if(temp_i!=pivot_row-1):
            multiplier=-float(A[temp_i][pivot_col-1])
            elementary_ops_change_row(A,temp_i+1,pivot_row,multiplier)        
        
        
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
    
    return A


def sweep_out_col_mult(
    A: np.array,
    ith_col: int,
    pivot_element: Tuple[int, int]
)->Tuple[np.array, np.array]:
    """
    Sweep out a i-th column based on the given pivot element
    
    Inputs:
        - A: Given matrix
        - ith_col: column to sweep out
        - pivot_element pivot element
        
    Output:
        - A: Resultant column sweep out matrix
        - E_ops: Product of all the elementary operations
    """
    
    E_ops = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    row,col = pivot_element
    m=len(A)
    n=len(A[0])
    E_ops=[[(1.0 if k==m_1 else 0.0)for k in range(m)]for m_1 in range(m)]
    scaling=1/float(A[row-1][col-1])
    elementary_ops_scale_row(E_ops,row,scaling)
    for p in range(m):
        if p!=row-1:
            I=[[(1.0 if k==m_2 else 0.0)for k in range(m)]for m_2 in range(m)]
            multiplier=-float(A[p][col-1])
            elementary_ops_change_row(I,p+1,row,multiplier)
            E_ops = matrix_multiplication(I,E_ops)
    A=matrix_multiplication(E_ops,A)
    
   
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
    
    return A, E_ops


def reduced_echelon_form(
    A: np.array
)-> np.array:
    """Reduction to echelon form of a matrix.
    
    Inputs:
        - A: Given matrix
        
    Output:
        - A: Echelon form of A
    """
    
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    m, n = len(A),len(A[0])
    temp = 0
    temp_r=[]
    for r in range(m):
        if temp >= n:
            break
        i = r

        while A[i][temp] == 0:
            i += 1
            if i == m:
                i = r
                temp += 1
                
                if temp == n:
                    return A
            A[[i, r]] = A[[r, i]]

            #elementary_ops_interchange_rows(A,r+1,i+1)
            #print(A)

        scaling = 1/float(A[r][temp])
        elementary_ops_scale_row(A,r+1,scaling)

        for i in range(m):
            if i != r:
                multiplier =-float(A[i][temp])
                elementary_ops_change_row(A,i+1,r+1,multiplier)
        temp += 1

                        
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A


def reduced_upper_triangular_form(
    A: np.array
)-> np.array:
    """Reduction to upper triangular form of a square matrix.
    
    Inputs:
        - A: Given matrix
        
    Output:
        - A: Upper triangular form of A
    """
    
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    row,col=len(A),len(A[0])
    for i in range(min(row, col)):
        if A[i][i] == 0:
            for k in range(i + 1, row):
                if A[k][i] != 0:
                    elementary_ops_interchange_rows(A,i+1,k+1)
                    break
        for j in range(i + 1, row):
            factor = float(A[j][i]) / float(A[i][i])
            elementary_ops_change_row(A,j+1,i+1,-factor)
        
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A


def reduce_non_sigular_matrix_to_identity(
    A: np.array
)-> Tuple[np.array, np.array]:
    """Reduce a non-singular matrix to a identity matrix.
    
    Inputs:
        - A: Given matrix
        
    Output:
        - I: Identity matrix
    """
    
    INV_A = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    _EPS = 1e-12

    j = 0
    k = 0
    n = len(A)
    if n == 0:
        raise ValueError("Input matrix A is empty.")
    if any(len(row) != n for row in A):
        raise ValueError("Input matrix A must be square (n x n).")

    INV_A = [[1.0 if p == r else 0.0 for p in range(n)] for r in range(n)]

    while j < n and k < n:
        # find a pivot in column j starting from row k (0-based)
        pivot_row = None
        for i in range(k, n):
            if abs(float(A[i][j])) > _EPS:
                pivot_row = i
                break

        if pivot_row is None:
            # no pivot in this column, move to next column
            j += 1
            continue

        # if pivot row is not the current k-th row, swap them (and update INV_A)
        if pivot_row != k:
            A, E_swap = elementary_ops_interchange_rows_mult(A, pivot_row + 1, k + 1)
            INV_A = matrix_multiplication(E_swap, INV_A)

        # now pivot is at row k (1-based = k+1). Sweep out column j.
        A, E_op = sweep_out_col_mult(A, j + 1, (k + 1, j + 1))
        INV_A = matrix_multiplication(E_op, INV_A)

        k += 1
        j += 1

        
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A, INV_A


def generate_random_linear_system(
    m: int,
    n: int,
    l: int = 9,
)->Tuple[np.array, np.array, np.array]:
    """
    Generate a random system of linear equation m-equation with n-unknown (with integer coefficients)
    
    Inputs:
        - m: Number of equations
        - n: Number of unknowns
        - l: range of random integers
        
    Outputs:
        - A: Coefficient matrix
        - b: right-hand vector
        - x: soluation (its just for cross check)
    """
    
    A = []
    b = []
    x = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    
    A=rand_matrix(m,n,l)
    b=rand_matrix(n,1,l)
    inv,ide=reduce_non_sigular_matrix_to_identity(A)
    x=matrix_multiplication(ide,b) 
    
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return A, b, x
    
    
def soln_n_equations_n_unknowns(
    A: np.array,
    b: np.array
)-> np.array:
    """Inverse of a non-singular matrix using reduced echelon form.
    
    Inputs:
        - A: Given coefficient matrix
        - b: Given right-hand vector
        
    Output:
        - x_est: Soluation of Ax=b
        - INV_A: Inverse of A
    """
        
    x_est = []
    INV_A = []
    ############################################################################
    #                             Your code will be here                       #
    #--------------------------------------------------------------------------#
    I_A,INV_A=reduce_non_sigular_matrix_to_identity(A)
    x_est=matrix_multiplication(INV_A,b)  
    
        
    #--------------------------------------------------------------------------#
    #                             End of your code                             #
    ############################################################################
        
    return x_est, INV_A

if __name__ == '__main__':  
    
    pass








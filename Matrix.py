# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:48:28 2023

@author: Dongli
"""
from copy import copy

class Matrix:
    
    '''
    init
    '''
    
    def __init__(self, matrix, is_equation_system = False):
        if is_equation_system and not self._validate_ls_matrix():
            raise ValueError('Matrix has to be a 3 x 4 array.')
        self.matrix = matrix
        self.height = len(self.matrix)
        self.width = len(self.matrix[0])        
        self.I = self._generate_i() # identity matrix
        
    '''
    Linear Equation Algorithms
    '''
   # ==========================================================================
   #   1. Sovle with Gaussian Elimination
   # ==========================================================================
    
    def _sort_matrix(self):
        '''
        1st core operation:
        sort matrix to avoid 0 at the diagonal (Interchanging rows)
        '''
        indexes = [n for n in range(self.height)]
        lookup = {}
        # the rows contains more 0 is prioritized
        matrix = sorted(self.matrix, key=lambda row:(row[:3].count(0)), reverse=True)
        for i in range(self.height):
           for j in range(self.width):
               if matrix[i][j] != 0 and j in indexes: # not zero and not sorted
                   lookup[j] = matrix[i] # assign row to dict with its index as the key
                   indexes.remove(j) # remove index from the list
                   break
        sorted_matrix = [lookup[k] for k in range(self.height)]
        self.matrix = sorted_matrix
    
    def _element_to_one(self, row_index, elem_index):
        '''
        2nd core operation:
        divide a number to turn the target element into 1 
        '''
        element = self.matrix[row_index][elem_index]
        if element == 0:
            print('\nWarning: Problem is diverted.')
            self.matrix = self.matrix[:2]
            self.print_matrix()
            raise ValueError(f'x{elem_index+1} is a free variable')
        new_row = [e/element for e in self.matrix[row_index]]
        self.matrix[row_index] = new_row
    
    def _element_to_zero(self, target_row_index, elem_index, assis_row_index):
        '''
        3rd core operation:
        # Apply -a*Rx + Ry to turn the target element into 0
        '''
        element = self.matrix[target_row_index][elem_index]
        combiner = [-a*element for a in self.matrix[assis_row_index]]
        new_row = [c+t for c,t in zip(combiner, self.matrix[target_row_index])]
        self.matrix[target_row_index] = new_row       
        
    def gaussian_elimination(self, print_steps = False):
        '''
        # main program function to solve matrix problems using the 3 core funcs
        # aka. row operations
        # assume after step 1, the matrix become: 
        # | a f i * |
        # | b d h * |
        # | c e g * |
        '''
        # ---------------------------
        # step 1: rearrange matrix
        self._sort_matrix()
        if print_steps:
            print('Step 1: Rm <-> Rn')
            self.print_matrix()
        # --------------------------
        # step 2: turn a into 1
        self._element_to_one(0, 0)
        if print_steps:
            print('\nStep 2: R1 <- R1/a')
            self.print_matrix()
        # --------------------------
        # step 3: turn b into 0
        self._element_to_zero(1, 0, 0)
        if print_steps:
            print('\nStep 3: R2 <- -bR1 + R2')
            self.print_matrix()
        # --------------------------
        # step 4: turn c into 0
        if self.height == 3:
            self._element_to_zero(2, 0, 0)
            if print_steps:
                print('\nStep 4: R3 <- -cR1 + R3')
                self.print_matrix()
        # --------------------------
        # step 5: turn d into 1
        self._element_to_one(1, 1)
        if print_steps:
            print('\nStep 5: R2 <- R2/d')
            self.print_matrix()
        # --------------------------
        # step 6: turn e into 0
        if self.height == 3:
            self._element_to_zero(2, 1, 1)
            if print_steps:
                print('\nStep 6: R3 <- -eR2 + R3')
                self.print_matrix()
        # --------------------------
        # step 7: turn f into 0
        self._element_to_zero(0, 1, 1)
        if print_steps:
            print('\nStep 7: R1 <- -fR2 + R1')
            self.print_matrix()
        # step 8: turn g into 1
        if self.height == 3:
            self._element_to_one(2, 2)
            if print_steps:
                print('\nStep 8: R3 <- R3/g')
                self.print_matrix()
        # --------------------------
        # step 9: turn h into 0
            self._element_to_zero(1, 2, 2)
            if print_steps:
                print('\nStep 9: R2 <- -hR3 + R2')
                self.print_matrix()
        # --------------------------
        # step 10: turn h into 0
            self._element_to_zero(0, 2, 2)
            if print_steps:
                print('\nStep 10: R1 <- -iR3 + R1')
                self.print_matrix()
            
        return self
    
   # ==========================================================================
   #   2. Solve with Inverse of Matrix   
   # ==========================================================================

    def inverse_solution(self, print_steps = False):
        '''
        For a Linear Equation System AX = B:
                |a11, a12, a13|
        Let A = |a21, a22, a23|, B = |b1,b2,b3|, X = |x1,x2,x3|
                |a31, a32, a33|
        --> X = (A^-1)B
        This function invokes minor, cofactor, determinant, and adjoint
        '''
        # prepare: split matrix
        A = [row[:-1] for row in self.matrix]
        A = Matrix(A)
        B = [[row[-1]] for row in self.matrix]
        B = Matrix(B)
        # step 1: calculate determinant
        det = A.det()
        # step 2: calculate adjoint
        adj = A.adjoint()
        # step 3: calculate inverse
        inv = adj.multiply(1/det)
        # step 4: calculate solution
        X = inv.multiply(B)
        # print steps
        if print_steps:
            print('Left hand matrix:')
            A.print_matrix()
            print('Right hand matrix:')
            B.print_matrix()
            print(f'1. determinant of A: {det}') 
            print('2. adjoint of A:')
            adj.print_matrix()
            print('3. inverse of A:') 
            inv.print_matrix()
            print('4. final solution:')
            X.print_matrix()
        return X
        
   # ==========================================================================
   #   3. Solve Linear Equation with Cramer’s Rule 
   # ==========================================================================

    def det_arrow_tech(self):
        # using arrow technique to calculate the determinant
        net = self._combine_matrix(Matrix([row[:-1] for row in self.matrix])).matrix
        start, from_left, from_right = 0, 0, 0
        while start < self.width: # sum the product of lines from left to right
            curr = [0, start]
            product = 1
            while curr[0] < self.height:
                product *= net[curr[0]][curr[1]]
                curr[0] += 1
                curr[1] += 1
            from_left += product
            start += 1
        start = self.width - 1
        while start < len(net[0]): # sum the product of lines from left to right
            curr = [0, start]
            product = 1
            while curr[0] < self.height:
                product *= net[curr[0]][curr[1]]
                curr[0] += 1
                curr[1] -= 1
            from_right += product
            start += 1
        det = from_left - from_right # subtracting to get determinant
        return det

    def _replace_col(self, col, col_matrix):
        # replce a col of the matrix with a column matrix
        # for example, in AX=B, replace target row of A with B
        if col_matrix.width != 1:
            raise ValueError('Not a column matrix!')
        A = [[row[i] for i in range(self.width)] for row in self.matrix]
        for i in range(self.height):
            A[i][col] = col_matrix.matrix[i][0]
        return Matrix(A)

    def cramers_rule(self, print_steps = False):
        '''
        This algorithm calculate Xn = det(An)/det(A)
        For practice, this function use Arrow Technique to calculate det
        '''
        # prepare: split matrix
        A = Matrix([row[:-1] for row in self.matrix])
        B = Matrix([[row[-1]] for row in self.matrix])
        det_A = A.det_arrow_tech()
        X = []
        for i in range(self.height):
            det_Ai = A._replace_col(i, B).det_arrow_tech()
            x = det_Ai / det_A
            X.append([x])
            if print_steps:
                print(f'step {i+1}: \ndet(A{i+1})={det_Ai}, det(A)={det_A}, x{i+1}={x}')
        return Matrix(X)
    
    '''
    Linear Equation Sovling and displaying solutions
    '''
           
    def solve_matrix(self, algorithm=None, print_steps = False):
        # aggregate the algorithoms
        if algorithm is None or algorithm == self.gaussian_elimination:
            return self.gaussian_elimination(print_steps)
        else:
            return algorithm(print_steps)          
    
    def display_solution(self, print_steps = False):
        # conduct a user friendly communication
        if print_steps:
            algo_index = input('What algorithm do you want to choose?\na: gaussian_elimination\nb: Inverse of Matrix\nc: Cramer’s Rule\n')
            match algo_index:
                case 'b':
                    solution = self.solve_matrix(self.inverse_solution, print_steps)
                case 'c':
                    solution = self.solve_matrix(self.cramers_rule, print_steps)
                case _:
                    solution = self.solve_matrix(self.gaussian_elimination, print_steps)    
        else:
            solution = self.solve_matrix(self.gaussian_elimination, print_steps)
        print(f'\nThe solution is: {[round(row[-1],2) for row in solution.matrix]}')

    '''
    Matrix operations
    '''
    def _is_addable(self, m1, m2):
        # check if two matrixes are addable
        if not all(isinstance(row, list) for row in m1) or not all(isinstance(row, list) for row in m2):
            return False
        if len(m1) != len(m2):
            return False
        if any(len(row) != len(m2[0]) for row in m1) or any(len(row) != len(m1[0]) for row in m2):
            return False
        return True
    
    def add(self, m2):
        # add 2 matrix objects and return a result matrix
        m1 = self.matrix
        if isinstance(m2, (int, float)): 
            # If one is scalar, redirect it to another function.
            m2 = Matrix(self.I)._scalar_multiply(m2).matrix
        else:
            m2 = m2.matrix
        if not self._is_addable(m1, m2):
            raise ValueError("The 2 matrixes are not addable")
        sum = [[0]*self.width for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                sum[i][j] = m1[i][j] + m2[i][j]
        sum_matrix = Matrix(sum)
        return sum_matrix
    
    def _is_multipliable(self, m1, m2):
        # check if two matrixes are multipliable
        if not all(isinstance(row, list) for row in m1) or not all(isinstance(row, list) for row in m2):
            return False
        if len(m1[0]) != len(m2):
            return False
        return True
    
    def multiply(self, m2):
        # multiply matrixes.  
        if isinstance(m2, (int, float)): 
            # If one is scalar, redirect it to another function.
            return self._scalar_multiply(m2)
        m1 = self.matrix
        m2 = m2.matrix
        if not self._is_multipliable(m1, m2):
            raise ValueError("The 2 matrixes are not multipliable")
        height = self.height
        width = len(m2[0])
        common = self.width
        # a size of the common row/col, aka, cols of A or rows of B
        product = [[0] * width for _ in range(height)]
        for i in range(height):
            for j in range(width):
                for c in range(common):
                    product[i][j] += m1[i][c]*m2[c][j]
        product_matrix = Matrix(product)
        return product_matrix
    
    def _scalar_multiply(self, scalar):
        # Calculate matrix x scalar
        product = [[0] * self.width for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                product[i][j] = self.matrix[i][j] * scalar
        product_matrix = Matrix(product)
        return product_matrix
       
    def minus(self, m2):
        # subtraction between matrices 
        if not self._is_addable(self.matrix, m2.matrix):
            raise ValueError("The 2 matrixes are not subtractable")
        neg =  [[-a for a in row] for row in m2.matrix]
        neg_m2 = Matrix(neg)
        return self.add(neg_m2)
    
    def trace(self):
        # get the trace of matrix. Aka., the sum in main diagonal
        if not self._is_squarematrix():
            raise ValueError('Not square Matrix. Not possible')
        trace = sum(self.matrix[i][i] for i in range(self.height))
        return trace
    
    def transpose(self):
        transposed = [[0] * self.height for _ in range(self.width)]
        transposed = [[self.matrix[j][i] for j in range(self.height)] for i in range(self.width)]
        return Matrix(transposed)
            
    def inverse(self):
        # calculate the inverse matrix
        if not self.is_invertable():
            raise ValueError("The matrix is not invertable.")
        return self._combine_matrix(Matrix(self.I)).solve_matrix()._right_half()
    
    '''
    
    Determinant, Joint, Minor, Cofactor 
    
    '''
    
    def det_2x2(self):
        # return determine of a 2x2 matrix
        if self.height != 2 or self.width!=2:
            raise ValueError("Not a 2x2 matrix!")
        return self.matrix[0][0] * self.matrix[1][1] - self.matrix[1][0] * self.matrix[0][1]
    
    def adjoint_2x2(self):
        # return adjoint of a 2x2 matrix or a minor of 3x3
        if self.height != 2 or self.width!=2:
            raise ValueError("Not a 2x2 matrix!")
        adj = [[self.matrix[1][1],-self.matrix[0][1]], 
             [-self.matrix[1][0], self.matrix[0][0]]]
        adjoint_matrix = Matrix(adj)
        return adjoint_matrix
    
    def is_invertable(self):
        # check if a matrix have a inverse or not
        if not self.height == self.width:
            return False
        return self.det()!=0
    
    def det(self):
        # calculate the determinant of a matrix
        if not self._is_squarematrix():
            raise ValueError("Only square matrix can have determinant")
        if self.width == 2:
            return self.det_2x2()
        # using Cofactor Expansion
        det = 0
        for i in range(self.width):
            minor_matrix = self._minor_matrix(0, i)
            minor = minor_matrix.det_2x2()
            cofactor = minor if self._on_diagonal(0, i) else -minor
            det += cofactor * self.matrix[0][i]
        return det
    
    def adjoint(self):
        # calculate the adjoint matrix of a 3x3 matrix
        if self.height == 2:
            return self.adjoint_2x2()
        ac = [[0] * self.width for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                minor = self._minor_matrix(i, j).det_2x2()
                cofactor = minor if self._on_diagonal(i, j) else -minor
                ac[i][j] = cofactor
        adj = Matrix(ac).transpose()
        return adj
        
    '''
    util funcs
    '''
            
    def _generate_i(self):
        # generate I according to the dimension of the self.matrix
        I = [[1 if i == j else 0 for j in range(self.width)] for i in range(self.height)]
        return I
    
    def _validate_ls_matrix(self):
        # check if the matrix has 3 x 4 list
        if isinstance(self.matrix, list) and len(self.matrix)==3:
            if all(isinstance(row, list) and len(row) == 4 for row in self.matrix):
                return True
        return False
      
    def _is_squarematrix(self):
        # check if the matrix is a square matrix
        return self.height == self.width
    
    def print_matrix(self):
        # Print matrix in a human friendly way
        for i in range(len(self.matrix)):
            print([round(num,2) for num in self.matrix[i]])
    
    def _combine_matrix(self, m2):
        # combine matrix so the width of the result will be wa + wb.
        m1 = self.matrix
        m2 = m2.matrix
        combined = [r1 + r2 for r1, r2 in zip(m1, m2)]        
        combined_matrix = Matrix(combined)
        return combined_matrix
    
    def _right_half(self):
        # get the right half of the matrix for 3x3(or more) inversion
        m = self.matrix
        width = self.width
        if width%2 != 0: # check wrong odd matrix
            raise ValueError('Odd width!')
        right = [row[width//2:] for row in m]
        right_matrix = Matrix(right)
        return right_matrix
    
    def _on_diagonal(self, row_index,col_index):
        # Check if aij is on the 2 diagonals of the matrix
        return row_index == col_index or row_index + col_index == self.height - 1
    
    def _minor_matrix(self, row_index, col_index):
        # return sub matrix that not have the same row index or col index
        minor = [[self.matrix[i][j] for j in range(self.width) if j!=col_index]
                 for i in range(self.width) if i!=row_index]
        return Matrix(minor)
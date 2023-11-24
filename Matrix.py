# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:48:28 2023

@author: Dongli
"""

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
        return row_index == col_index or row_index + col_index == self.height - 1
    
    def _minor_matrix(self, row_index, col_index):
        minor = [[self.matrix[i][j] for j in range(self.width) if j!=col_index]
                 for i in range(self.width) if i!=row_index]
        return Matrix(minor)
    
    '''
    Linear Equation Algorithms
    '''
# =============================================================================
#     Gaussian Elimination
# =============================================================================
    
    def _sort_matrix(self):
        '''
        1st core function:
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
        2nd core function:
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
        3rd core function:
        # Apply -a*Rx + Ry to turn the target element into 0
        '''
        element = self.matrix[target_row_index][elem_index]
        combiner = [-a*element for a in self.matrix[assis_row_index]]
        new_row = [c+t for c,t in zip(combiner, self.matrix[target_row_index])]
        self.matrix[target_row_index] = new_row       
        
    def gaussian_elimination(self, print_steps = False):
        '''
        # main program function to solve matrix problems using the 3 core funcs
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
    
    '''
    Linear Equation Sovling and displaying solutions
    '''
           
    def solve_matrix(self, algorithm=None, print_steps = False):
        # aggregate the algorithoms
        if algorithm is None or algorithm == self.gaussian_elimination:
            return self.gaussian_elimination(print_steps)
        else:
            algorithm(self, print_steps);
            
    
    def display_solution(self, algorithm=None, print_steps = False):
        # print a read friendly solution
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
        if not self.is_squarematrix():
            raise ValueError("Only square matrix can have determinand")
        # using 
        
        
        
        
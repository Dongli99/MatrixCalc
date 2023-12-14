# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:48:28 2023

@author: Dongli
"""
from math import sqrt, acos, cos, radians, degrees, asin

class Matrix:
    
    '''
    init
    '''
    
    def __init__(self, matrix):
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
        try:
            self._sort_matrix()
        except Exception:
            raise ValueError('Too many 0s, try another algorithm.')
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
           
    def solve_matrix(self, algorithm=None, print_steps=False, strict=False):
        # aggregate the algorithoms
        if strict and not self._is_equation_matrix():
            raise ValueError("The matrix must be a 3x4 matrix.")
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
            raise ValueError('Not square Matrix. Not possible for trace')
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
    
    def _is_equation_matrix(self):
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
    
class Vector:
    '''
    init
    '''
    def __init__(self, head, tail=[0,0]):
        self.head = head
        self.zero = [0] * len(self.head)
        if tail==[0,0] and len(self.head) > 2: # handle length
            self.tail = self.zero
        else:
            self.tail = tail
        self.vector = [ h-t for h,t in zip(self.head,self.tail)]
        self.magnitude = self.normalize()
        self.dimension = len(self.vector)
        
    '''
    utils
    '''    
    def print_vector(self): # print vector as a list
        print([round(v, 2) for v in self.vector])
        
    def is_equal_to(self, vector): # check if the 2 vectors are identical
        return all(v == u for v,u in zip(self.vector,vector.vector))
    
    def is_nagative_to(self, vector): # check if the 2 vectors are opposite
        return all(v == -u for v,u in zip(self.vector,vector.vector))

    def get_negtive(self): # given v, get -v
        return Vector(self.zero).minus(self)    
    
    def normalize(self): # Ecliden norm of the vector
        return sqrt(sum(pow(value, 2) for value in self.vector))
        
    def get_unit_vector(self): # get the unit vector(v^)
        return self.scalar_multiply(1/self.normalize())
    
    def is_unit_vector(self): # check if the vector is a unit vector
        return self.unit_vector().is_equal_to(self)
    
    def distance(a, b, c, x0, y0):
        # this is a special func to calc distance of (x0, y0) to ax+by+c=0
        return abs(a*x0 + b*y0 + c)/sqrt(a*a + b*b)

    '''
    operations
    '''
    
    def add(self, vector): # addition
        res = [v + w for v,w in zip(self.vector, vector.vector)]
        return Vector(res)
    
    def minus(self, vector): # subtraction
        res = [v - w for v,w in zip(self.vector, vector.vector)]
        return Vector(res) 
    
    def scalar_multiply(self, a): # times a scalar
        res = [a * v for v in self.vector]
        return Vector(res)

    def midpoint(self, vector): # get the midpoint of 2 vectors
        return self.add(vector).scalar_multiply(1/2)

    def dot_product(self, vector): # dot product using components
        return sum([v * w for v,w in zip(self.vector, vector.vector)])
    
    def dot_product_cos(v_norm, u_norm, angle): # dot product using angle
        return v_norm * u_norm * cos(radians(angle))
    
    def angle(self, vector): # get degree of angle between u and v
        return degrees(acos(self.dot_product(vector)/(
            self.normalize()*vector.normalize())))
    
    def angle_cross_prod(self, vector): # get angle using cross product
        return degrees(asin(self.cross_product(vector).normalize()/(
            self.normalize()*vector.normalize())))
    
    def is_perpendicular(self, vector): 
        # check if 2 vectors are perpendicular
        return self.dot_product(vector) == 0
    
    def is_perp_set(self, vector1, vector2): 
        # identify perpendicular set
        return (self.is_perpendicular(vector1) and 
                self.is_perpendicular(vector2) and 
                vector1.is_perpendicular(vector2))
    
    def proj_along(self, vector): # projection(self)vector
        return vector.scalar_multiply(
            self.dot_product(vector)/vector.dot_product(vector))
    
    def proj_orth(self, vector): # projection along self
        return self.minus(self.proj_along(vector))
    
    def cross_product(self, vector): # cross product of vectors
        m0 = Matrix([[self.vector[i] for i in range(3) if i!=0],
              [vector.vector[j] for j in range(3) if j!=0]])
        m1 = Matrix([[self.vector[i] for i in range(3) if i!=1],
              [vector.vector[j] for j in range(3) if j!=1]])
        m2 = Matrix([[self.vector[i] for i in range(3) if i!=2],
              [vector.vector[j] for j in range(3) if j!=2]])
        return Vector([m0.det(), -m1.det(), m2.det()])
    
    '''
    geometrical significance
    '''
    
    def parallelogram_area(self, vector): # parallelogram_area of 2 vectors
        if self.dimension == 2:
            return abs(Matrix([self.vector, vector.vector]).det())
        return self.cross_product(vector).normalize()
    
    def triangle_area(self, vector): # triangle_area of 2 vectors
        return self.parallelogram_area(vector) / 2
    
    def triangle_scalar(self, v, u): 
        # retrun the trangle product scalar of 3 vectors
        return self.dot_product(v.cross_product(u))
    
    def triangle_scalar_det(self, v, u): 
        # retrun the trangle product scalar using det algorithm
        return Matrix([self.vector, v.vector, u.vector]).det()
    
    def parallelepiped_vol(self, v, u): # volume of parallelepiped
        return abs(self.triangle_scalar(v, u))
    
    def tetrahedron_vol(self, v, u): # volume of tetrahedron
        return self.parallelepiped_vol(v, u)/3
    
    def are_coplanar(self, v, u): # check if 3 vectors on the same plane
        return self.triangle_scalar(v, u) == 0
        
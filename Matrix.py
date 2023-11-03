# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:48:28 2023

@author: Dongli
"""

class Matrix:
    '''
    init and basic functions
    '''
    
    def __init__(self, matrix):
        if self._validate_matrix(matrix):
            self.matrix = matrix
        else:
            raise ValueError('Matrix has to be a 3 x 4 array.')
            
    def _validate_matrix(self, matrix):
        # check if the matrix has 3 x 4 list
        if isinstance(matrix, list) and len(matrix)==3:
            if all(isinstance(row, list) and len(row) == 4 for row in matrix):
                return True
        return False
    
    def _normalize_matrix(self, matrix):
        # Transform lines that contains 0 in the first 3 elements
        for i in range(3):
            if 0 in matrix[i][:3]:
                matrix[i] = [v1 + v2 for v1, v2 in zip(matrix[i], matrix[i-1])]
        return matrix
    
    def _print_matrix(self):
        for i in range(3):
            print(self.matrix[i])
   
    '''
    3 core calculation algorithms
    '''
    
    def _sort_matrix(self):
        orders = [0,1,2]
        lookup = {}
        matrix = sorted(self.matrix, key=lambda row:(row.count(0)), reverse=True)
        for i in range(3):
           for j in range(3):
               if matrix[i][j] != 0 and j in orders:
                   lookup[j] = matrix[i]
                   orders.remove(j)
                   break
        sorted_matrix = []
        sorted_matrix = [lookup[k] for k in range(3)]
        self.matrix = sorted_matrix
    
    def _element_to_one(self, row_index, elem_index):
        # divide a number to turn the target element into 1 
        element = self.matrix[row_index][elem_index]
        new_row = [e/element for e in self.matrix[row_index]]
        self.matrix[row_index] = new_row
    
    def _element_to_zero(self, target_row_index, elem_index, assis_row_index):
        # Apply -a*Rx + Ry to turn the target element into 0
        element = self.matrix[target_row_index][elem_index]
        combiner = [-a*element for a in self.matrix[assis_row_index]]
        new_row = [c+t for c,t in zip(combiner, self.matrix[target_row_index])]
        self.matrix[target_row_index] = new_row
    
    '''
    Program
    '''
    
    def solve_matrix(self, print_steps = False):
        # main program function to solve matrix problems
        # assume after step 1, the matrix become: 
        # | a f i * |
        # | b d h * |
        # | c e g * |
        
        # ---------------------------
        # step 1: rearrange matrix
        self._sort_matrix()
        if print_steps:
            print('Step 1:')
            self._print_matrix()
        # --------------------------
        # step 2: turn a into 1
        self._element_to_one(0, 0)
        if print_steps:
            print('Step 2:')
            self._print_matrix()
        # --------------------------
        # step 3: turn b into 0
        self._element_to_zero(1, 0, 0)
        if print_steps:
            print('Step 3:')
            self._print_matrix()
        # --------------------------
        # step 4: turn c into 0
        self._element_to_zero(2, 0, 0)
        if print_steps:
            print('Step 4:')
            self._print_matrix()
        # --------------------------
        # step 5: turn d into 1
        self._element_to_one(1, 1)
        if print_steps:
            print('Step 5:')
            self._print_matrix()
        # --------------------------
        # step 6: turn e into 0
        self._element_to_zero(2, 1, 1)
        if print_steps:
            print('Step 6:')
            self._print_matrix()
        # --------------------------
        # step 7: turn f into 0
        self._element_to_zero(0, 1, 1)
        if print_steps:
            print('Step 7:')
            self._print_matrix()
        # step 8: turn g into 1
        self._element_to_one(2, 2)
        if print_steps:
            print('Step 8:')
            self._print_matrix()
        # --------------------------
        # step 9: turn h into 0
        self._element_to_zero(1, 2, 2)
        if print_steps:
            print('Step 9:')
            self._print_matrix()
        # --------------------------
        # step 10: turn h into 0
        self._element_to_zero(0, 2, 2)
        if print_steps:
            print('Step 10:')
            self._print_matrix()      

        # Return answer
        return [row[-1] for row in self.matrix]
        
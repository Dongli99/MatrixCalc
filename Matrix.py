# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:48:28 2023

@author: Dongli
"""

class Matrix:
    
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
        new_row = [c*t for c,t in zip(combiner, self.matrix[target_row_index])]
        self.matrix[target_row_index] = new_row
        
    def solve_matrix(self):
        pass

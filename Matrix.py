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
   
    '''
    calculation algorithms
    '''
    
    def sort_matrix(self, matrix):
        orders = [0,1,2]
        lookup = {}
        matrix = sorted(matrix, key=lambda row:(row.count(0)))
        for i in range(3):
           for j in range(3):
               if matrix[i][j] != 0 and j in orders:
                   lookup[j] = matrix[i]
                   orders.remove(j)
        sorted_matrix = []
        sorted_matrix = [lookup[k] for k in range(3)]
        return sorted_matrix
    
    def _element_to_one(self, row, index):
        return [e/row[index] for e in row]
    
    def 
       
    def solve_matrix(self):
        pass

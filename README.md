# MatrixCalc

- [MatrixCalc](#matrixcalc)
  - [Overview](#overview)
  - [How to use](#how-to-use)
    - [Before use](#before-use)
    - [Declare a Matrix Object](#declare-a-matrix-object)
    - [Recommended approach](#recommended-approach)
    - [Solve a Linear Equation System](#solve-a-linear-equation-system)
      - [Only display the answer](#only-display-the-answer)
      - [Choose algorithms and display detailed steps](#choose-algorithms-and-display-detailed-steps)
    - [Matrix Operations](#matrix-operations)
      - [Inverse](#inverse)
      - [Transpose](#transpose)
      - [Trace](#trace)
      - [Addition](#addition)
      - [Subtraction](#subtraction)
      - [Multiply](#multiply)
    - [Determinant and Adjoint](#determinant-and-adjoint)
      - [Determinant](#determinant)
      - [Adjoint](#adjoint)

## Overview

***Purpose*** - This project is designed to solve Matrix problems. The purpose of this project is to practice and enhance the knowledge of both Linear Algebra and Python.  
***Updating*** - New features will be added along study of Linear Algebra.

## How to use

### Before use

- `git clone https://github.com/Dongli99/MatrixCalc.git`
- Create a python script under MatrixCalc/
- on the top of the script, paste ```from Matrix import Matrix```

### Declare a Matrix Object  

Syntax - `A = Matrix(a,is_equation_system=False)`

```python
a = [[1,2,3],
    [4,5,6],
    [7,8,9]]
A = Matrix(a)
```

### Recommended approach

- Chain the methods is recommended.

Example 1: $((C^{-1})^T)*{10}$

```python
# Example 1: ((C^-1)^T)10
c = [[3,-1],
     [4,2]]
res = C.inverse().transpose().multiply(10)
res.print_matrix()
```

```python
# output
[2.0, -4.0]
[1.0, 3.0]
```

Example 2:
 $AC(AC)^{-1}(DA^{-1})^{-1}(AD^T)^TA^T(B^{-1}A^T)^{-1}B^{-1}$

```python
# After declaring any 4 metrics (better square metrics)
res = A.multiply(C).multiply(A.multiply(C).inverse()).multiply(D.multiply(A.inverse()).inverse()).multiply(A.multiply(D.transpose()).transpose()).multiply(A.transpose()).multiply(B.inverse().multiply(A.transpose()).inverse()).multiply(B.inverse())
# print result
res.print_matrix()
```

```python
# output. After simplifying, the formula turns to be a symmetric Matrix
[20.0, -36.0, -8.0]
[-36.0, 85.0, 7.0]
[-8.0, 7.0, 6.0]
```

### Solve a Linear Equation System  

Syntax - `A.display_solution(print_steps=False) -> void`

#### Only display the answer

```python
a = [[1,-2,4,1],
     [3,-1,4,-1],
     [-3,-1,1,8]]
A = Matrix(a)
A.display_solution()
```

```python
# output
The solution is: [-3.0, 4.0, 3.0]
```

#### Choose algorithms and display detailed steps

```python
A.display_solution(True)
```

```python
# display algorithm options
What algorithm do you want to choose?
a: gaussian_elimination
b: Inverse of Matrix
c: Cramer’s Rule
```

```python
# if choose 'a' or any entry other than 'b' and 'c'
# output of gaussian_elimination algorithm
# also known as row operations
Step 1: Rm <-> Rn
[1, -2, 4, 1]
[3, -1, 4, -1]
[-3, -1, 1, 8]

Step 2: R1 <- R1/a
[1.0, -2.0, 4.0, 1.0]
[3, -1, 4, -1]
[-3, -1, 1, 8]

Step 3: R2 <- -bR1 + R2
[1.0, -2.0, 4.0, 1.0]
[0.0, 5.0, -8.0, -4.0]
[-3, -1, 1, 8]

...

Step 10: R1 <- -iR3 + R1
[1.0, 0.0, 0.0, -3.0]
[0.0, 1.0, 0.0, 4.0]
[0.0, 0.0, 1.0, 3.0]

The solution is: [-3.0, 4.0, 3.0]
```

```python
# if choose 'b'
# output calculation using inverse approach
# the algorithm involves determinant, cofactor, adjoint...
Left hand matrix:
[1, -2, 4]
[3, -1, 4]
[-3, -1, 1]
Right hand matrix:
[1]
[-1]
[8]
1. determinant of A: 9
2. adjoint of A:
[3, -2, -4]
[-15, 13, 8]
[-6, 7, 5]
3. inverse of A:
[0.33, -0.22, -0.44]
[-1.67, 1.44, 0.89]
[-0.67, 0.78, 0.56]
4. final solution:
[-3.0]
[4.0]
[3.0]

The solution is: [-3.0, 4.0, 3.0]
```

```python
# if choose 'c'
# output the steps using Cramer’s Rule
# this rule uses arrow technique and xn = det(An)/det(A)

step 1: 
det(A1)=-27, det(A)=9, x1=-3.0
step 2: 
det(A2)=36, det(A)=9, x2=4.0
step 3: 
det(A3)=27, det(A)=9, x3=3.0

The solution is: [-3.0, 4.0, 3.0]
```

### Matrix Operations

#### Inverse

Syntax - `A.inverse() -> Matrix`

```python
# Inverse of a 2x2 matrix
a = [[1,-2],
     [3,-1]]
A = Matrix(a).inverse()
A.print_matrix()
# print_matrix() prints matrix with good appearance
```

```python
# output
[-0.2, 0.4]
[-0.6, 0.2]
```

```python
# Inverse of a 3X3 matrix
a = [[1,-2, 5],
     [3,-1, 6],
     [4, 5, 9]]
A = Matrix(a).inverse()
A.print_matrix()  
```

```python
# output
[-0.63, 0.69, -0.11]
[-0.05, -0.18, 0.15]
[0.31, -0.21, 0.08]
```

#### Transpose

Syntax - `A.transpose() -> Matrix`

```python
a = [[1, -1, -2],
     [1, 1, 3],
     [2, -1, 1]]
A = Matrix(a)
A.transpose().print_matrix()
```

```python
# output
[1, 1, 2]
[-1, 1, -1]
[-2, 3, 1]
```

#### Trace

Syntax - `A.trace() -> Matrix`

```python
a = [[1,-2, 5],
     [3,-1, 6],
     [4, 5, 9]]
A = Matrix(a)
tr = A.trace()
```

```python
# output a scalar
9
# if height != width, output
ValueError: Not square Matrix. Not possible
```

#### Addition

Syntax - `A.add(B) -> Matrix`

```python
a = [[1,-2, 5],
     [3,-1, 6],
     [4, 5, 9]]
b = [[6,-1, 5],
     [6,-1, 6],
     [4, 3, 1]]
A = Matrix(a)
B = Matrix(b)
C = A.add(B)
C.print_matrix()
```

```python
# output
[7, -3, 10]
[9, -2, 12]
[8, 8, 10]
```

```python
# if A and B have different dimensions
# output
ValueError: The 2 matrixes are not addable
```

#### Subtraction

Syntax - `A.minus(B) -> Matrix`

```python
a = [[1,-2, 5],
     [3,-1, 6],
     [4, 5, 9]]
b = [[6,-1, 5],
     [6,-1, 6],
     [4, 3, 1]]
A = Matrix(a)
B = Matrix(b)
C = A.minus(B)
C.print_matrix()
```

```python
# output
[-5, -1, 0]
[-3, 0, 0]
[0, 2, 8]

ValueError: The 2 matrixes are not subtractable
```

#### Multiply

Syntax - `A.multiply(B) -> Matrix`

```python
## Matrix x Matrix
C = A.multiply(B)
C.print_matrix()
```

```python
# output
[14, 16, -2]
[36, 16, 15]
[90, 18, 59]

# If A_col != B_row, output
ValueError: The 2 matrixes are not multiplicable
```

```python
## Matrix x scalar
C = A.multiply(5)
C.print_matrix()
```

```python
# output
[5, -10, 25]
[15, -5, 30]
[20, 25, 45]
```

### Determinant and Adjoint

#### Determinant

Syntax - `A.det() -> float/int`

```python
# Can handel 2x2 and 3x3 matrices
a = [[3, 2, -1],
     [1, 6, 3],
     [2, -4, 0]]
A = Matrix(a)
det = A.det()
print(det)
```

```python
# output
[12, 6, -16]
[4, 2, 16]
[12, -10, 16]
```

#### Adjoint

Syntax - `A.adjoint() -> Matrix`

```python
# Can handel 2x2 and 3x3 matrices
adj = A.adjoint()
adj.print_matrix()
```

```python
# output
[12, 6, -16]
[4, 2, 16]
[12, -10, 16]
```

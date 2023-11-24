# MatrixCalc

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

#### Display detailed steps

```python
A.display_solution(True)
```

```python
# output
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

Syntax - `A.det() -> float`

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

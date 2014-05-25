*
* maximize +10 X1 +20 X2 +100 X3 +50 X4 +5 X5 +1 X6
*
* 1 X1 +1 X2 +1 X3 +1 X4             <= 3
* 1 X1 +1 X2 +2 X3 +1 X4             <= 2
*             1 X3 +1 X4 +1 X5 +1 X6 <= 2
*             3 X3 +1 X4 +2 X5 +1 X6 <= 5
*


NAME          DEMO5
ROWS
 N  OBJECTIVE
 L  C1
 L  C2
 L  C3
 L  C4
COLUMNS
    X1      OBJECTIVE         10   C1                 1
    X1      C2                 1
    X2      OBJECTIVE         20   C1                 1
    X2      C2                 1
    X3      OBJECTIVE        100   C1                 1
    X3      C2                 2   C3                 1
    X3      C4                 3
    X4      OBJECTIVE         50   C1                 1
    X4      C2                 1   C3                 1
    X4      C4                 1
    X5      OBJECTIVE          5   C3                 1
    X5      C4                 2
    X6      OBJECTIVE          1   C3                 1
    X6      C4                 1
RHS
    RHS1    C1                 3   C2                 2
    RHS1    C3                 2   C4                 5
BOUNDS
 UP BND1    X1                 1
 LO BND1    X1                 0
 UP BND1    X2                 1
 LO BND1    X2                 0
 UP BND1    X3                 1
 LO BND1    X3                 0
 UP BND1    X4                 1
 LO BND1    X4                 0
 UP BND1    X5                 1
 LO BND1    X5                 0
 UP BND1    X6                 1
 LO BND1    X6                 0
ENDATA

*
* maximize +2 X1 +3 X2 +5 X3 +1 X4 +4 X5
*
* 1 X1 +1 X2 +1 X3             <= 2
*             1 X3 +1 X4 +1 X5 <= 2
*


NAME          DEMO8
ROWS
 N  OBJECTIVE
 L  C1
 L  C2
COLUMNS
    X1      OBJECTIVE          2   C1                 1
    X2      OBJECTIVE          3   C1                 1
    X3      OBJECTIVE          5   C1                 1
    X3      C2                 1
    X4      OBJECTIVE          1   C2                 1
    X5      OBJECTIVE          1   C2                 1
RHS
    RHS1    C1                 2   C2                 2
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
ENDATA

#
# You need to have the packages below (and only those below)
# installed with your Python interpreter
# 
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from random import randrange

iii = np.zeros(1000)
for ii in range(1000): 
    N = 5
    #
    # This function generates a random square matrix M
    # of size intN x IntN, s.t., for each column 
    # (1) there is a single 1 in any row i in [0..IntN-1], or
    # (2) the whole column contains 0s
    # where each of the above happens uniformly at random
    # with the probability 1/(IntN+1)
    #
    def GenerateRandomMatrix(intN):

        T = [0 for i in range(intN)]
        for j in range (intN):
            T[j]= randrange(-1, intN)
        
        M = [0 for i in range(intN)]
        for j in range (intN):
            M[j]= [0 for i in range(intN)]

        for i in range(intN):
            for j in range(intN):
                if (T[j] == i):
                    M[i][j] = 1
                else:
                    M[i][j] = 0
        return M

    # 
    # This function is expected to compute the rank of matrix MX
    # of size len(MX) x len (MX) generated earlier by GenerateRandomMatrix
    # Thus MX is in a very specific format
    #
    # =============== EDIT below this line ================
    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    #
    def RankMatrix (MX):

        rank = 0
        Size=len(MX)

        column = np.zeros(Size)
        for i in range(Size):
            for j in range(Size):
                if MX[i][j] == 1:
                    column[i] = 1
        for i in range(Size):
            rank = rank + column[i]
        rank = int(rank)
        return(rank)

        # MX_np = np.array(MX)
        # rank1 = np.linalg.matrix_rank(MX)


    # 
    # Write above the missing part which computes the rank of MX
    # You can use only loops and direct operations on MX entries
    # !!! USE OF ANY EXTRA PREDEFINED PACKAGES IN PYTHON IS DISALLOWED !!!
    # Note! The current version always reports the rank 0
    # [WORTH 5 POINTS]

        # return(rank)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # =============== EDIT above this line ================
    #

    #
    # Compute the mean of the discrete probability mass distribution
    # in which value i comes with the probability X[i], for i in [0..len(X)-1] 
    #
    # =============== EDIT below this line ================
    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

    def Mean (X):

        MN = 0
        Size = len(X)
        
        for i in range(Size):
            MN = MN + X[i] * i

        return MN    

    #
    # Write here the missing part which computes the mean described above
    # You can use only loops and direct operations on X entries
    # !!! USE OF ANY EXTRA PREDEFINED PACKAGES IN PYTHON IS DISALLOWED !!!
    # Note! The current version always reports the mean 0
    # [WORTH 3 POINTS]

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # =============== EDIT above this line ================
    #

    #
    # Read from the input the size N of the random matrix
    # Accept sizes between 1 and 199
    #


    #
    # RM becomes a random matrix of size N x N
    #
    RM = GenerateRandomMatrix(N)

    #
    # Print the computed (by RankMatrix) rank of RM
    # 
    #
    print ("The rank of the random matrix is: ", RankMatrix(RM))


    #
    # Repeat N*N times computation of random matrix RM and
    # collect the statistics about the frequencies of the ranks  
    # Store the statistics in array Stats
    #
    Stats = [0.0 for i in range(N+1)]
    Attempts = N**2 # the number of attempts

    for j in range(Attempts):
        RM = GenerateRandomMatrix(N)
        rk = RankMatrix(RM)
        Stats[rk] = Stats[rk] + 1.0

    #
    # Replace the frequency statistics with
    # the discrete probablity mass distribution (also in Stats)
    # and print the mean value
    #
    Max = 0
    PosMax = 0
    for j in range(N+1):
        Stats[j] = Stats[j]/Attempts    
        if (Stats[j] > Max):
            Max = Stats[j]
            PosMax = j

    MeanValue = Mean(Stats)
    print("The computed mean value is: ", MeanValue)
    iii[ii] = MeanValue

    #
    # Display the probability mass distribution
    # The vertical blue line denotes the mean of the distribution
    #   
    ax = plt.subplot()
    ax.set_xlim([-1,N+1])
    ax.set_ylim([-0.1,Max+0.1])

    xg = []
    yg = []

    for i in range (N):
        xg.extend([i])
        yg.extend([Stats[i]])

    gr=ax.scatter(xg,yg,color='green')

    ax.vlines(MeanValue,0,Max+0.1)
    print(ii)

    # plt.draw()

    # plt.show()
    # plt.savefig('./test.jpg')

    #
    # DO NOT FORGET to answer questions Q1 an Q2 in
    # a separate document PDF document. 
    # Make sure the answers are brief.
    # [WORTH 2 x 1 POINT]
#
print(iii.mean()/N)
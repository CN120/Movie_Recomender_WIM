import numpy as np
from math import log



trained = np.loadtxt("./train.txt", dtype=int, delimiter='\t')


print(trained)

test5 = np.loadtxt("test5.txt", dtype=int, delimiter=' ')
test10 = np.loadtxt("test10.txt", dtype=int, delimiter=' ')
test20 = np.loadtxt("test20.txt", dtype=int, delimiter=' ')

testing = None

def buildTesting(test_num):
    global testing
    testing = np.zeros([100,1000], dtype=int)
    if test_num == 5:
        test_arr = test5
    elif test_num == 10:
        test_arr = test10
    else:
        test_arr = test20
    
    first_item = test_arr[0][0]
    for row in test_arr:
        testing[row[0]-first_item, row[1]-1] = row[2] 


def printNP(arr):
    for item in arr:
        print(item)

#will modify trained array
def computeIUF():
    global trained
    m = 200
    for col in range(trained.shape[1]):
        nz = np.count_nonzero(trained[:,col])
        if nz!=0:
            trained[:,col] = trained[:,col]* (log(m)-log(nz))

#-------------
#similarities
#-------------
def computeCosSim(user1, user2):
    #expects user to be 1 dimmensional ndarray
    mask = np.logical_and(user1, user2)
    a = user1[mask]
    b = user2[mask]
    if a.size <=1:
        return a.size
    else:
        return np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))

def computePearsonSim(user1, user2):
    mask = np.logical_and(user1, user2)
    a = user1[mask]
    b = user2[mask]
    if a.size <= 1:
        return a.size
    else:
        avg1 = np.average(a)
        avg2 = np.average(b)
        a = a-avg1
        b = b-avg2
        denom = (np.linalg.norm(a)*np.linalg.norm(b))
        if denom==0:
            return 0
        weight =  round(np.sum(a*b)/denom, 10)
        return weight #modification 2 when commented out
        p=2.5
        new_weight = weight * (abs(weight)**(p-1))
        return new_weight

def computeCosIBCFSim(item1, item2):
    s1 = item1.size 
    s2 = item2.size
    if s1>s2:
        item1 = item1[:s2]
    else:
        item2 = item2[:s1]
    if s1 <=1:
        return s1
    else:
        avg1 = np.average(item1)
        avg2 = np.average(item2)
        item1 = item1-avg1
        item2 = item2-avg2
        denom = (np.linalg.norm(item1)*np.linalg.norm(item2))
        if denom==0:
            return 0
        return round(np.sum(item1*item2)/denom,10)


#-------------        
#predictors
#-------------
def predictBasicCosUBCF(userID, movieID, k=None):
    neighbors = np.zeros([200,2],dtype=np.float32)
    filled = 0
    userID = (userID-1)%100
    movieID = movieID-1
    for row in trained:
        if row[movieID]==0:
            continue
        else:
            sim = computeCosSim(row,testing[userID]) 
            if sim>0:
                neighbors[filled][0] = sim
                neighbors[filled][1] = row[movieID]
                filled+=1
    sorted_knn = (neighbors[np.argsort(neighbors[:,0])])[::-1]
    # printNP(sorted_knn)
    if k==None:
        knns = sorted_knn[:filled-1] #swapped k for filled-1
    else:
        knns = sorted_knn[:k] #swapped k for filled-1
    numer = np.sum(knns[:,0]*knns[:,1])
    denom = np.sum(knns[:,0])  #vertical sum of weights
    if denom>0:
        prediction = round(numer/denom)
        if prediction>5:
            return 5
        return prediction
    else:
        return 3

def predictBasicPearsonUBCF(userID, movieID, k=None):
    neighbors = np.zeros([200,2],dtype=np.float32)
    filled = 0
    userID = (userID-1)%100
    movieID = movieID-1
    for row in trained:
        if row[movieID]==0:
            continue
        else:
            sim = computePearsonSim(testing[userID], row) #compare active user with each row/user from trained
            if abs(sim)>1:
                print(sim)
            if sim!=0:
                neighbors[filled][0] = sim  #first column contains similarity rating
                neighbors[filled][1] = row[movieID] #second column contains the ranking given to movie by 
                filled+=1
    sorted_knn = (neighbors[np.argsort(abs(neighbors[:,0]))])[::-1]
    # printNP(sorted_knn)
    # print("---------------------")
    if k==None:
        knns = sorted_knn[:filled-1] #swapped k for filled-1
    else:
        knns = sorted_knn[:k] #swapped k for filled-1
    avg_u = np.average(knns[:,1])
    numer = np.sum(knns[:,0]*(knns[:,1]-avg_u))
    denom = np.sum(abs(knns[:,0]))  #vertical sum of weights

    ra = np.sum(testing[userID])/np.count_nonzero(testing[userID])
    # print(ra)
    if denom!=0:
        prediction = round(ra+(numer/denom))
        if prediction>5:
            return 5
        if prediction<=0:
            return 1
        return prediction
    else:
        return 3

def predictCosIBCF(userID, movieID, k=None):
    neighbors = np.zeros([1000,2],dtype=np.float32)
    filled = 0
    userID = (userID-1)%100
    movieID = movieID-1
    for i in range(trained.shape[1]):
        col = trained[:,i]
        if col[userID]==0:
            continue
        else:
            sim = computeCosIBCFSim(col,testing[:,movieID]) 
            if sim>0:
                neighbors[filled][0] = sim
                neighbors[filled][1] = col[userID]
                filled+=1
    sorted_knn = (neighbors[np.argsort(neighbors[:,0])])[::-1]
    # printNP(sorted_knn)
    if k==None:
        knns = sorted_knn[:filled-1] #swapped k for filled-1
    else:
        knns = sorted_knn[:k] #swapped k for filled-1
    numer = np.sum(knns[:,0]*knns[:,1])
    denom = np.sum(knns[:,0])  #vertical sum of weights
    if denom>0:
        prediction = round(numer/denom)
        if prediction>5:
            return 5
        return prediction
    else:
        return 3

def predictCustom(userID, movieID, k=None):
    neighbors = np.zeros([200,2],dtype=np.float32)
    filled = 0
    userID = (userID-1)%100
    movieID = movieID-1
    for row in trained:
        if row[movieID]==0:
            continue
        else:
            sim = computeCosSim(row,testing[userID]) 
            if sim>0:
                neighbors[filled][0] = sim
                neighbors[filled][1] = row[movieID]
                filled+=1
    sorted_knn = (neighbors[np.argsort(neighbors[:,0])])[::-1]
    # printNP(sorted_knn)
    if k==None:
        knns = sorted_knn[:filled] #swapped k for filled-1
    else:
        knns = sorted_knn[:k] #swapped k for filled-1
    numer = 0.9 * np.sum(knns[:,0]*knns[:,1]) + (0.1 * (np.sum(trained[:,movieID])/np.count_nonzero(trained[:,movieID])))
    denom = np.sum(knns[:,0])  #vertical sum of weights
    if denom>0:
        prediction = round(numer/denom)
        if prediction>5:
            return 5
        return prediction
    else:
        return 3
#--------------
#drivers
#--------------
def basicCosUBCF():
    global test5, test10, test20
    buildTesting(5)
    test5 = test5[np.any(test5 == 0, axis=1)]
    # printNP(test5)
    for row in test5:
        row[2] = predictBasicCosUBCF(row[0],row[1])
    print(test5)
    np.savetxt("result5.txt", test5,fmt='%d', delimiter=' ')

    buildTesting(10)
    test10 = test10[np.any(test10 == 0, axis=1)]
    for row in test10:
        row[2] = predictBasicCosUBCF(row[0],row[1])
    print(test10)
    np.savetxt("result10.txt", test10,fmt='%d', delimiter=' ')

    buildTesting(20)
    test20 = test20[np.any(test20 == 0, axis=1)]
    for row in test20:
        row[2] = predictBasicCosUBCF(row[0],row[1])
    print(test20)
    np.savetxt("result20.txt", test20,fmt='%d', delimiter=' ')

def basicPearsonUBCF():
    global test5, test10, test20
    # computeIUF()  #modification 1 when uncommented (& mod 2 is commented)

    buildTesting(5) #sets test
    test5 = test5[np.any(test5 == 0, axis=1)]
    # printNP(test5)
    for row in test5:
        row[2] = predictBasicPearsonUBCF(row[0],row[1])
    print(test5)
    np.savetxt("result5.txt", test5,fmt='%d', delimiter=' ')

    buildTesting(10)
    test10 = test10[np.any(test10 == 0, axis=1)]
    for row in test10:
        row[2] = predictBasicPearsonUBCF(row[0],row[1])
    print(test10)
    np.savetxt("result10.txt", test10,fmt='%d', delimiter=' ')

    buildTesting(20)
    test20 = test20[np.any(test20 == 0, axis=1)]
    for row in test20:
        row[2] = predictBasicPearsonUBCF(row[0],row[1])
    print(test20)
    np.savetxt("result20.txt", test20,fmt='%d', delimiter=' ')

def cosIBCF():
    global test5, test10, test20
    buildTesting(5)
    test5 = test5[np.any(test5 == 0, axis=1)]
    # printNP(test5)
    for row in test5:
        row[2] = predictCosIBCF(row[0],row[1])
    print(test5)
    np.savetxt("result5.txt", test5,fmt='%d', delimiter=' ')

    buildTesting(10)
    test10 = test10[np.any(test10 == 0, axis=1)]
    for row in test10:
        row[2] = predictCosIBCF(row[0],row[1])
    print(test10)
    np.savetxt("result10.txt", test10,fmt='%d', delimiter=' ')

    buildTesting(20)
    test20 = test20[np.any(test20 == 0, axis=1)]
    for row in test20:
        row[2] = predictCosIBCF(row[0],row[1])
    print(test20)
    np.savetxt("result20.txt", test20,fmt='%d', delimiter=' ')



def custom():
    global test5, test10, test20
    # computeIUF()
    buildTesting(5)
    test5 = test5[np.any(test5 == 0, axis=1)]
    # printNP(test5)
    for row in test5:
        row[2] = predictCustom(row[0],row[1])
    print(test5)
    np.savetxt("result5.txt", test5,fmt='%d', delimiter=' ')

    buildTesting(10)
    test10 = test10[np.any(test10 == 0, axis=1)]
    for row in test10:
        row[2] = predictCustom(row[0],row[1])
    print(test10)
    np.savetxt("result10.txt", test10,fmt='%d', delimiter=' ')

    buildTesting(20)
    test20 = test20[np.any(test20 == 0, axis=1)]
    for row in test20:
        row[2] = predictCustom(row[0],row[1])
    print(test20)
    np.savetxt("result20.txt", test20,fmt='%d', delimiter=' ')


# basicCosUBCF()
# basicPearsonUBCF()
# cosIBCF()
custom()
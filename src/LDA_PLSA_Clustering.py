import pyodbc
import numpy as np
from numpy.core.fromnumeric import shape
import numpy as np

from sklearn.decomposition import LatentDirichletAllocation
import PLSA as plsa
from SOM.SelfOrganisingMaps import SelfOrganisingMaps
from BatAlgorithm import *
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

from timeit import default_timer as timer   


__Matrix = []

def Unique(List):
     unique_list = []     
     for x in List:
          if x not in unique_list:
               unique_list.append(x)    
     return unique_list

def ExecuteQuery(Query: str):
    ConnectionString = "Server=.;Database=News_Recommender_By_SOM;UID=sa;PWD=bb@@11"
    conn = pyodbc.connect('Driver={SQL Server};' + ConnectionString, autocommit=True)
    cursor = conn.cursor()
    Response = []
    try:
        query = "SET NOCOUNT ON; " + Query
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        for row in cursor.fetchall():
            Response.append(dict(zip(columns, row[:])))    
    except pyodbc.Error as ex:
        print(ex)
        print(Query)
        pass
    finally:
        cursor.close()
        del cursor
        conn.close()
    return Response

def GetData(top):
    DataRaw = ExecuteQuery(f"SELECT TOP {top} ID, Features F, Target T FROM dbo.DataForTrain")
    Data = []
    for row in DataRaw:
        try:
            X = list(map(int, row["F"].split(",")))
            Y = list(map(int, row["T"].split(",")))

            if(len(X) < 500):
                for i in range(500 - len(X)):
                    X.append(0)
            else :
                X = X[0:500]
            if(len(Y) < 5):
                for i in range(5 - len(Y)):
                    Y.append(0)
            #print(shape(X), shape(Y))
            item = [X,Y]
            Data.append(item)
        except:
            pass	
    return Data

def GetLAD(Input):
    Data = []
    X = []    
    for item in Input:
        X.append(item[0])    
    print(shape(X))
    lda = LatentDirichletAllocation(n_components=100, random_state=0)
    lda.fit(X)
    Data = lda.transform(X)
    return Data

def GetPLSA(Input):
    Res = []
    _plsa = plsa.PLSATrain(50)
    trainedTopics = _plsa.train(Input, 0.001)
    myPLSA = plsa.PLSATest(50)
    Res = myPLSA.query(Input, 0.001)

    return Res

def SOM_Clustering(Matrix, Clusters):

    MatrixSize = len(Matrix)
    BatchSize = len(Matrix)
    SOM = SelfOrganisingMaps(Clusters, MatrixSize, 10)

    SOM.Fit(Matrix)

    Unique_Labels = Unique(SOM.Labels)
    
    if (len(Unique_Labels) == 1):
        if(SOM.Labels[0] == 0):
            SOM.Labels[0] = 1
        else:
            SOM.Labels[0] = 0
        
    _silhouette_score = silhouette_score(Matrix, SOM.Labels, metric="euclidean")
    _bouldin_score = davies_bouldin_score(Matrix, SOM.Labels)
    _calinski_harabasz_score = calinski_harabasz_score(Matrix, SOM.Labels)

    print("For n_clusters =", Clusters, "The silhouette_score is :", _silhouette_score)
    print("For n_clusters =", Clusters, "The bouldin_score is :", _bouldin_score, "lower Is better")
    print("For n_clusters =", Clusters, "The calinski_harabasz_score is :", _calinski_harabasz_score, "higher Is better")

    print("End Of SOM Clustering")
    return 0

def BAT_Fitness_Function(D, sol):
    # print("__Matrix :", __Matrix)
    # print("sol :", sol)
    labels = np.array(sol).astype(int)
    for i in range(len(labels)):
        if(labels[i] < 0):
            labels[i] = 0
    # print("labels :", labels)
    Unique_Labels = Unique(labels)
    if (len(Unique_Labels) == 1):
        if(labels[0] == 0):
            labels[0] = 1
        else:
            labels[0] = 0

    _silhouette_score = silhouette_score(__Matrix, labels, metric="euclidean")
    _silhouette_score = _silhouette_score * -1
    
    # print("_silhouette_score :", _silhouette_score)

    return _silhouette_score

def SOM_BAT_Clustering(Matrix, Clusters):

    MatrixSize = len(Matrix)
    BatchSize = len(Matrix)
    SOM = SelfOrganisingMaps(Clusters, MatrixSize, 10)

    SOM.Fit(Matrix)

    # print(SOM.Labels)

    Bat = BatAlgorithm(MatrixSize, 10, 100, 0.5, 0.5, 0.0, 1.0, 1.0, Clusters, BAT_Fitness_Function)
    Bat.best = SOM.Labels
    for i in range(5):
        Bat.move_bat()
    SOM.Labels = Bat.best

    # print(SOM.Labels)

    Unique_Labels = Unique(SOM.Labels)
    
    if (len(Unique_Labels) == 1):
        if(SOM.Labels[0] == 0):
            SOM.Labels[0] = 1
        else:
            SOM.Labels[0] = 0
        
    _silhouette_score = silhouette_score(Matrix, SOM.Labels, metric="euclidean")
    _bouldin_score = davies_bouldin_score(Matrix, SOM.Labels)
    _calinski_harabasz_score = calinski_harabasz_score(Matrix, SOM.Labels)

    print("For n_clusters =", Clusters, "The silhouette_score is :", _silhouette_score)
    print("For n_clusters =", Clusters, "The bouldin_score is :", _bouldin_score, "lower Is better")
    print("For n_clusters =", Clusters, "The calinski_harabasz_score is :", _calinski_harabasz_score, "higher Is better")

    print("End Of SOM_BAT Clustering")
    
    return 0

def DBSCAN_Clustering(Matrix, Clusters):

    dbscan_clustering = DBSCAN(eps=3, min_samples=Clusters).fit(Matrix)
    
    Unique_Labels = Unique(dbscan_clustering.labels_)
    
    if (len(Unique_Labels) == 1):
        if(dbscan_clustering.labels_[0] == 0):
            dbscan_clustering.labels_[0] = 1
        else:
            dbscan_clustering.labels_[0] = 0
        
    _silhouette_score = silhouette_score(Matrix, dbscan_clustering.labels_, metric="euclidean")
    _bouldin_score = davies_bouldin_score(Matrix, dbscan_clustering.labels_)
    _calinski_harabasz_score = calinski_harabasz_score(Matrix, dbscan_clustering.labels_)

    print("For n_clusters =", Clusters, "The silhouette_score is :", _silhouette_score)
    print("For n_clusters =", Clusters, "The bouldin_score is :", _bouldin_score, "lower Is better")
    print("For n_clusters =", Clusters, "The calinski_harabasz_score is :", _calinski_harabasz_score, "higher Is better")

    print("End Of DBSCAN Clustering")
    return 0

    return 0

def Main():
    
    start = timer()
    Data = GetData(9500)
    print("GetData: ", timer()-start)    

    start = timer()
    Lda = GetLAD(Data)
    print(shape(Lda))
    print("GetLAD: ", timer()-start)  

    start = timer()
    Plsa = GetPLSA(Lda)
    print("GetPLSA: ", timer()-start)  

    global __Matrix
    __Matrix = Plsa

    # SOM_Clustering(Plsa, 3)
    # SOM_Clustering(Plsa, 5)
    # SOM_Clustering(Plsa, 7)
    # SOM_Clustering(Plsa, 9)
    # SOM_Clustering(Plsa, 11)
    # SOM_Clustering(Plsa, 13)
    # SOM_Clustering(Plsa, 15)

    start = timer()
    SOM_BAT_Clustering(Plsa, 3)
    print("SOM_BAT_Clustering 3: ", timer()-start)  
    
    start = timer()
    SOM_BAT_Clustering(Plsa, 5)
    print("SOM_BAT_Clustering 5: ", timer()-start)  
    
    start = timer()
    SOM_BAT_Clustering(Plsa, 7)
    print("SOM_BAT_Clustering 7: ", timer()-start)  

    start = timer()
    SOM_BAT_Clustering(Plsa, 9)
    print("SOM_BAT_Clustering 9: ", timer()-start)  

    start = timer()
    SOM_BAT_Clustering(Plsa, 11)
    print("SOM_BAT_Clustering 11: ", timer()-start)  

    start = timer()
    SOM_BAT_Clustering(Plsa, 13)
    print("SOM_BAT_Clustering 13: ", timer()-start)  

    start = timer()
    SOM_BAT_Clustering(Plsa, 15)
    print("SOM_BAT_Clustering 15: ", timer()-start)  
    
    # DBSCAN_Clustering(Plsa, 3)
    # DBSCAN_Clustering(Plsa, 5)
    # DBSCAN_Clustering(Plsa, 7)
    # DBSCAN_Clustering(Plsa, 9)
    # DBSCAN_Clustering(Plsa, 11)
    # DBSCAN_Clustering(Plsa, 13)
    # DBSCAN_Clustering(Plsa, 15)

Main()
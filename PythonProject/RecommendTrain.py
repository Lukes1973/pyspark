#-*- coding:UTF-8-*-

import os
import sys
from pyspark import SparkContext
from pyspark import SparkConf
from blaze.expr.core import path

from pyspark.mllib.recommendation import Rating,MatrixFactorizationModel
from pyspark.mllib.recommendation import ALS

# reload(sys)
# sys.setdefaultencoding('utf-8')


def CreateSparkContext():
    sparkConf = SparkConf().setAppName("RecommendTrains").set("spark.ui.showConsoleProgress","false")
    sc = SparkContext(conf=sparkConf)
    print("master="+sc.master)
    SetLogger(sc)
    SetPath(sc)
    return(sc)

def SetLogger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("log").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)
    
def SetPath(sc):
    global path
    if sc.master[0:5]=="local":
        path = "file:/home/hduser/pythonwork/PythonProject/"
    else:
        path="hdfs://master:9000/user/hduser/"


def SaveModel(sc):
    try:
        model.save(sc,path+"ALSmodel")
        print("  已存储 Model 在 ALSmodel")
    except Exception:
        print("Model  已经存在，请先删除再存储。")
        
        
def PrepareData(sc):
     rawUserData = sc.textFile(path + "data/u.data")
     rawRatings = rawUserData.map(lambda line:line.split("\t")[:3])
     ratingsRDD = rawRatings.map(lambda x:(x[0],x[1],x[2]))
     return ratingsRDD
     
     
if __name__ == '__main__':
    sc=CreateSparkContext()
    print("=========== 数据准备阶段 ============")
    ratingsRDD = PrepareData(sc)
    print("=========== 训练阶段 ================")
    print(" 开始 ALS 训练，参数 rank =5 ,iterations =20 ,lambda = 0.1")
    model = ALS.train(ratingsRDD,5,20,0.1)
    print("============存储 Model =============")
#     print( model.recommendProducts(100, 5))
#     MatrixFactorizationModel.load(model)
#     print(model)
    SaveModel(sc)
    
    
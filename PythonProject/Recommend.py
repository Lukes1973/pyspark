#-*- coding:UTF-8-*-


import os
import sys
from pyspark import SparkContext
from pyspark import SparkConf
from blaze.expr.core import path
global model

# reload(sys)
# sys.setdefaultencoding('utf-8')

from pyspark.mllib.recommendation import Rating,ALS,MatrixFactorizationModel


def CreateSparkContext():
    sparkConf = SparkConf().setAppName("Recommend").set("spark.ui.showConsoleProgress","false")
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



def PrepareData(sc):
    print("开始读取电影ID与名称字典")
    itemRDD = sc.textFile(path+"data/u.item")
    movieTitle=itemRDD.map(lambda line:line.split("|")).map(lambda a:(float(a[0]),a[1])).collectAsMap()
    return (movieTitle)
 
def Recommend(model):
    if sys.argv[1]=="U":
         RecommendMovies(model,movieTitle,int(sys.argv[2]))
    if sys.argv[1]=="M":
         RecommendUsers(model,movieTitle,int(sys.argv[2]))
        
def RecommendMovies(model,movieTitle,inputUserID):
    RecommendMovies = model.recommendProducts(inputUserID,10)
    print(" 针对用户id" + str(inputUserID) + "推荐下列电影")
    for rmd in RecommendMovies:
        print ("针对用户id{0}推荐电影{1}推荐评分{2}".format(rmd[0],movieTitle[rmd[1]],rmd[2]))
        
        
def RecommendUsers(model,movieTitle,inputMovieID):
    RecommendUser = model.recommendUsers(inputMovieID,10)
    print ("针对电影id{0} 电影名:{1}推荐下列用户id:".format(inputMovieID,movieTitle[inputMovieID]))
    for rmd in RecommendUser:
        print ("针对用户id {0} 推荐评分{1}".format(rmd[0],rmd[2]))
        
def loadModel(sc):
#     model = MatrixFactorizationModel.load(sc,path+"ALSmodel")
    try:
        model = MatrixFactorizationModel.load(sc,path+"ALSmodel")
        print("载入ALSModel模型")
    except Exception:
        print("找不到模型")
    return model

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("请输入2个参数")
        exit(-1)
    sc=CreateSparkContext()
    print("=========== 数据准备 =============")
    (movieTitle)=PrepareData(sc)
    print("=========== 载入模型 =============")
    model =loadModel(sc)
    print("============进行推荐=============")
    Recommend(model)
#     print(sys.argv[2])
    
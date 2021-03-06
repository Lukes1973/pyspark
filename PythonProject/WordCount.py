#-*- coding:UTF-8-*-
import os
import sys
from pyspark import SparkContext
from pyspark import SparkConf
from blaze.expr.core import path

# reload(sys)
# sys.setdefaultencoding('UTF-8')


def CreateSparkContent():
    sparkConf = SparkConf().setAppName("WordCounts").set("spark.ui.showConsoleProgress","false")
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
        
        
if __name__ == '__main__':
    print(" 开始执行 RunWordCount")
    sc = CreateSparkContent()
    print ("开始读取文本文件....")
    textFile = sc.textFile(path+"data/README.md")
    print("    文本文件共 " + str(textFile.count()) + "  行 ")

    countsRDD = textFile.flatMap(lambda line:line.split('  ')).map(lambda x: (x,1)).reduceByKey(lambda x,y :x+y)
    print("  文字统计共  "+str(countsRDD.count()) + " 项数据")

    print("  开始保存文件至文本文件 ....")
    try:
        countsRDD.saveAsTextFile(path + "data/output")
    except Exception as e:
        print(" 输出目录已经存在,请先删除原有目录 ")
    sc.stop()
    
    
    
    

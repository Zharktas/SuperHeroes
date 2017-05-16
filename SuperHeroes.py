from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import csv
from itertools import combinations

from graphframes import *
from pyspark.sql.types import *
from pyspark.sql.functions import desc

spark = SparkSession  \
    .builder \
    .appName("SuperHeroes") \
    .getOrCreate()

sc = spark.sparkContext
data = sc.textFile("hero-comic-network.csv")


edge_data = data.map(lambda row: next(csv.reader(row.splitlines(), skipinitialspace=True)))\
    .map(lambda x: (x[1], x[0])).distinct()


listOfConnections = edge_data.groupByKey()\
    .mapValues(list)\
    .map(lambda x: x[1])

sortedListOfConnections = listOfConnections.map(lambda x: sorted(x))


connectionPairs = sortedListOfConnections.flatMap(lambda x: list(combinations(x, 2)))

connectionPairCount = connectionPairs.map(lambda x: (x, 1)).reduceByKey(lambda a,b: a+b).map(lambda x: (x[0][0], x[0][1], x[1]))

heroes = edge_data.map(lambda x: (x[1],)).distinct()



vertexSchema = StructType([StructField("id", StringType(), False)])
vertices = spark.createDataFrame(heroes, vertexSchema)

edgeSchema = StructType([StructField("src", StringType(), False),
                         StructField("dst", StringType(), False),
                         StructField("connections", IntegerType(), False)])

edges = spark.createDataFrame(connectionPairCount, edgeSchema)

graph = GraphFrame(vertices, edges)

graph.degrees.sort("degree", ascending=False).show()
graph.degrees.filter("id == 'CAPTAIN AMERICA'").show()

sc.setCheckpointDir(".")
results = graph.connectedComponents()
results.filter("component != 0").write.save("disconnected", format="csv")

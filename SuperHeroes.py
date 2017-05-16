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

#data = ['"FROST, CARMILLA","AA2 35"', '"KILLRAVEN/JONATHAN R","AA2 35"']

#heroes = sc.parallelize(data)

edge_data = data.map(lambda row: next(csv.reader(row.splitlines(), skipinitialspace=True)))\
    .map(lambda x: (x[1], x[0])).distinct()

#edge_data.saveAsTextFile("flipped")

combined = edge_data.groupByKey()\
    .mapValues(list)\
    .map(lambda x: x[1])

sorted_list = combined.map(lambda x: sorted(x))

#combined.saveAsTextFile("combined")

pairs = sorted_list.flatMap(lambda x: list(combinations(x, 2)))

edge_count = pairs.map(lambda x: (x, 1)).reduceByKey(lambda a,b: a+b).map(lambda x: (x[0][0], x[0][1], x[1]))
#pairs.saveAsTextFile("cartesian")



heroes = edge_data.map(lambda x: (x[1],)).distinct()
#comics = edge_data.map(lambda x: (x[1], "comic"))
#formatted_data = heroes.union(comics)



vertexSchema = StructType([StructField("id", StringType(), False)])
vertices = spark.createDataFrame(heroes, vertexSchema)

edgeSchema = StructType([StructField("src", StringType(), False),
                         StructField("dst", StringType(), False),
                         StructField("connections", IntegerType(), False)])

edges = spark.createDataFrame(edge_count, edgeSchema)

graph_with_connections = GraphFrame(vertices, edges)


edge_without_connectionsSchema = StructType([StructField("src", StringType(), False),
                         StructField("dst", StringType(), False)])
edges_without_connections = spark.createDataFrame(pairs, edge_without_connectionsSchema)

graph_without_connections = GraphFrame(vertices, edges_without_connections)

pagerank_with_connections = graph_with_connections.pageRank(resetProbability=0.15, tol=0.01)
pagerank_with_connections.vertices.sort(desc("pagerank")).limit(100).write.save("pagerank_with_connections", format="csv")

pagerank_without_connections = graph_without_connections.pageRank(resetProbability=0.15, tol=0.01)
pagerank_without_connections.vertices.sort(desc("pagerank")).limit(100).write.save("pagerank_without_connections", format="csv")

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.clustering import KMeans

# Create Spark session
spark = SparkSession.builder.appName("CarInsuranceKMeans").getOrCreate()

# Load car claim data
car_data = spark.read.csv("data/CIC2.csv", header=True, inferSchema=True)

# Prepare the features for K-means clustering
feature_columns = ['credit_score', 'vehicle_ownership', 'married', 'children',
			'speeding_violations', 'duis', 'past_accidents', 'outcome',
                   '16-25', '26-39', '40-64', '65+', 'female', 'male', '0-9y',
                   '10-19y', '20-29y', '30y+', 'high school', 'none',
                   'university', 'middle class', 'poverty', 'upper class',
                   'working class', 'after 2015', 'before 2015', 'sedan',
                   'sports car']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
car_data = assembler.transform(car_data)

# Train K-means model
d = 2  # Replace with the desired number of clusters
kmeans = KMeans(k=d, seed=123)
model = kmeans.fit(car_data)

# Save the model
model.save("kmeans_model")

spark.stop()

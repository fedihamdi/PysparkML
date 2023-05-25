from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeansModel

# Create Spark session
spark = SparkSession.builder.appName("CarInsuranceKMeans").getOrCreate()


# Load the model
loaded_model = KMeansModel.load("kmeans_model")

# Load new customer data for prediction
new_data = spark.read.csv("CIC3.csv", header=True, inferSchema=True)

# Prepare features for prediction
feature_columns = ['credit_score', 'vehicle_ownership', 'married', 'children',
			'speeding_violations', 'duis', 'past_accidents', 'outcome',
                   '16-25', '26-39', '40-64', '65+', 'female', 'male', '0-9y',
                   '10-19y', '20-29y', '30y+', 'high school', 'none',
                   'university', 'middle class', 'poverty', 'upper class',
                   'working class', 'after 2015', 'before 2015', 'sedan',
                   'sports car']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
new_data = assembler.transform(new_data)

# Make predictions using the loaded model
predictions = loaded_model.transform(new_data)

# Extract the predicted insurance premiums
predicted_premiums = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

# Print the predicted insurance premiums
for premium in predicted_premiums:
    print("###\n"*10)
    if premium == 1:
        print("Predicted Insurance Premium Product: Medium", premium,"\n")
        print("Predicted Insurance Premium Price: €", 1000*1.75,"\n")
    elif premium == 2:
        print("Predicted Insurance Premium Product: Large", premium,"\n")
        print("Predicted Insurance Premium Price: €", 1000*2.5,"\n")
    else:
        print("Predicted Insurance Premium Product: Small", premium,"\n")
        print("Predicted Insurance Premium Price: €", 1000,"\n")
    print("###\n"*10)

# Stop the Spark session
spark.stop()

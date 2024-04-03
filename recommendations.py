import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType, StructField
from pyspark.sql.functions import col, explode

class MovieRecommender:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName('Movie Recommendations with ALS') \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")

    def load_data(self, ratings_file_path, movies_file_path):
        ratings_schema = StructType([
            StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("rating", FloatType(), True),
            StructField("timestamp", StringType(), True)
        ])
        
        movies_schema = StructType([
            StructField("movieId", IntegerType(), True),
            StructField("title", StringType(), True),
            StructField("genres", StringType(), True)
        ])
        
        ratings_df = self.spark.read.csv(ratings_file_path, header=True, schema=ratings_schema)
        movies_df = self.spark.read.csv(movies_file_path, header=True, schema=movies_schema)
        return ratings_df, movies_df

    def train_als_model(self, ratings_df):
        als = ALS(
            maxIter=5,
            regParam=0.01,
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop"
        )
        return als.fit(ratings_df)

    def recommend_movies(self, model, movies_df, user_id, num_recommendations):
        user_df = self.spark.createDataFrame([(user_id,)], ['userId'])
        recommendations = model.recommendForUserSubset(user_df, num_recommendations)
        
        recommendations = recommendations.select(
            col("userId"), explode(col("recommendations")).alias("recommendation")
        ).select(
            col("userId"), 
            col("recommendation.movieId").alias("movieId"),
            col("recommendation.rating").alias("rating")
        )
        
        return recommendations.join(movies_df, recommendations.movieId == movies_df.movieId) \
            .select('userId', 'title', 'rating')

    def run(self, movie_input_file_path, ratings_input_file_path, user_id, num_recommendations):
        ratings_df, movies_df = self.load_data(ratings_input_file_path, movie_input_file_path)
        model = self.train_als_model(ratings_df)
        user_recommendations = self.recommend_movies(model, movies_df, user_id, num_recommendations)
        
        user_recommendations.show(truncate=False)
        
        output_csv_path = '../data/output/recommendations'
        user_recommendations.coalesce(1).write.csv(output_csv_path, mode="overwrite", header=True)
        
        self.spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: program.py <movie_input_file_path> <ratings_input_file_path> <user_id> <num_recommendations>")
        sys.exit(1)

    recommender = MovieRecommender()
    recommender.run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

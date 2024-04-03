import csv
import random
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql.functions import col, avg, desc, count
from pyspark.ml.recommendation import ALS

def generate_movie_data(num_movies):
    return [ [movie_id, f"Movie {movie_id}", random.choice(["Action", "Comedy", "Drama", "Thriller", "Sci-Fi"])]
             for movie_id in range(1, num_movies + 1) ]

def generate_rating_data(num_users, num_movies, max_ratings_per_user):
    ratings_data = []
    for user_id in range(1, num_users + 1):
        num_ratings = random.randint(1, max_ratings_per_user)
        rated_movies = random.sample(range(1, num_movies + 1), num_ratings)
        for movie_id in rated_movies:
            ratings_data.append([user_id, movie_id, random.randint(1, 5)])
    return ratings_data

def write_to_csv(file_name, data, header):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

def create_spark_session():
    return SparkSession.builder.appName("CombinedAnalysis").getOrCreate()

def read_data(spark, file_name, schema):
    return spark.read.csv(file_name, header=True, schema=schema).dropna(how='any')

def basic_analysis(movies_df, ratings_df):
    total_movies = movies_df.count()
    total_ratings = ratings_df.count()
    avg_ratings_per_movie = ratings_df.groupBy("movieId").agg(avg("rating").alias("avg_rating"))
    top_rated_movies = avg_ratings_per_movie.join(movies_df, "movieId").orderBy(desc("avg_rating")).select("title", "avg_rating").limit(10)

    return total_movies, total_ratings, avg_ratings_per_movie, top_rated_movies

def recommendation_analysis(ratings_df, num_recommendations, user_id):
    als = ALS(maxIter=10, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(ratings_df)
    user_recommendations = model.recommendForAllUsers(num_recommendations).filter(col("userId") == user_id)
    recommended_movie_ids = [row.movieId for row in user_recommendations.collect()[0].recommendations]
    return recommended_movie_ids

def additional_analysis(ratings_df, movies_df):
    genre_avg_ratings = ratings_df.join(movies_df, "movieId").groupBy("genre").agg(avg("rating").alias("avg_rating"))
    genre_rating_counts = ratings_df.join(movies_df, "movieId").groupBy("genre").agg(count("rating").alias("rating_count")).orderBy(desc("rating_count"))
    user_movie_counts = ratings_df.groupBy("userId").agg(count("movieId").alias("movie_count")).orderBy(desc("movie_count"))

    return genre_avg_ratings, genre_rating_counts, user_movie_counts

def main():
    num_movies = 200
    num_users = 100
    max_ratings_per_user = 30

    movies_data = generate_movie_data(num_movies)
    ratings_data = generate_rating_data(num_users, num_movies, max_ratings_per_user)

    write_to_csv("custom_movies.csv", movies_data, ["movieId", "title", "genre"])
    write_to_csv("custom_ratings.csv", ratings_data, ["userId", "movieId", "rating"])

    spark = create_spark_session()
    movies_schema = StructType([ StructField("movieId", IntegerType()), StructField("title", StringType()), StructField("genre", StringType()) ])
    ratings_schema = StructType([ StructField("userId", IntegerType()), StructField("movieId", IntegerType()), StructField("rating", FloatType()) ])

    movies_df = read_data(spark, "custom_movies.csv", movies_schema)
    ratings_df = read_data(spark, "custom_ratings.csv", ratings_schema)

    total_movies, total_ratings, avg_ratings_per_movie, top_rated_movies = basic_analysis(movies_df, ratings_df)
    recommended_movie_ids = recommendation_analysis(ratings_df, 5, 1)
    genre_avg_ratings, genre_rating_counts, user_movie_counts = additional_analysis(ratings_df, movies_df)

    print(f"Total number of movies: {total_movies}")
    print(f"Total number of ratings: {total_ratings}")
    avg_ratings_per_movie.show()
    top_rated_movies.show()

    print(f"Top 5 movie recommendations for user 1: {recommended_movie_ids}")

    genre_avg_ratings.show(truncate=False)
    genre_rating_counts.show(truncate=False)
    user_movie_counts.show(truncate=False)

    while True:
        pass
    spark.stop()

if __name__ == "__main__":
    main()

import os
from pyspark import SparkConf, SparkContext

app_conf = SparkConf().setAppName("GenreAvgRating").setMaster("local[*]")
sc = SparkContext(conf=app_conf)
sc.setLogLevel("ERROR")

INPUT_DIR  = "hdfs://localhost:9000/movie/input"
PATH_MOVIES = f"{INPUT_DIR}/movies.txt"
PATH_RATE1  = f"{INPUT_DIR}/ratings_1.txt"
PATH_RATE2  = f"{INPUT_DIR}/ratings_2.txt"

HDFS_OUT  = "hdfs://localhost:9000/movie/output/bai2"
LOCAL_OUT = "/mnt/d/UIT 3rd year/BigData/ThucHanh/Lab3/output/bai2.txt"

# --- Bước 1: Tạo map movieId → danh sách thể loại ---
def parse_genres(line):
    parts = line.split(",")
    genres = parts[2].strip().split("|")
    return (int(parts[0]), genres)

movie_genres = sc.textFile(PATH_MOVIES).map(parse_genres).collectAsMap()
genres_bc = sc.broadcast(movie_genres)

# --- Bước 2: Mỗi rating → (genre, (rating, 1)) cho tất cả thể loại của phim đó ---
def expand_by_genre(cols):
    mid = int(cols[1])
    rating = float(cols[2])
    return [(g, (rating, 1)) for g in genres_bc.value.get(mid, [])]

rating_cols = sc.textFile(f"{PATH_RATE1},{PATH_RATE2}") \
                .map(lambda line: line.split(","))

genre_pairs = rating_cols.flatMap(expand_by_genre)

# --- Bước 3: Tính trung bình theo thể loại ---
def calc_avg(record):
    genre, (total, count) = record
    return (genre, round(total / count, 4), count)

def add_pairs(a, b):
    return (a[0] + b[0], a[1] + b[1])

result = genre_pairs.reduceByKey(add_pairs) \
                    .map(calc_avg) \
                    .sortBy(lambda r: -r[1]) \
                    .collect()

# --- In kết quả ---
header  = f"\n  {'STT':<5} {'Thể loại':<25} {'Avg Rating':>10} {'Số lượt':>10}"
divider = "  " + "-" * 52
print(header)
print(divider)
for i, (genre, avg, cnt) in enumerate(result, 1):
    print(f"  {i:<5} {genre:<25} {avg:>10.4f} {cnt:>10}")

# --- Ghi file local ---
output_lines = [header, divider]
for i, (genre, avg, cnt) in enumerate(result, 1):
    output_lines.append(f"  {i:<5} {genre:<25} {avg:>10.4f} {cnt:>10}")

# --- Lưu HDFS ---
sc.parallelize(result) \
  .map(lambda r: f"{r[0]}::{r[1]}::{r[2]}") \
  .saveAsTextFile(HDFS_OUT)
print(f"\n>>> Đã lưu kết quả lên HDFS: {HDFS_OUT}")

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)
with open(LOCAL_OUT, "w", encoding="utf-8") as fout:
    fout.write("\n".join(output_lines))
print(f">>> Đã lưu kết quả về local: {LOCAL_OUT}")
print("=" * 60 + "\n")

sc.stop()

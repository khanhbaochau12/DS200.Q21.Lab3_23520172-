import os
from pyspark import SparkConf, SparkContext

app_conf = SparkConf().setAppName("MovieAvgRating").setMaster("local[*]")
sc = SparkContext(conf=app_conf)
sc.setLogLevel("ERROR")

INPUT_DIR   = "hdfs://localhost:9000/movie/input"
PATH_MOVIES  = f"{INPUT_DIR}/movies.txt"
PATH_RATE1   = f"{INPUT_DIR}/ratings_1.txt"
PATH_RATE2   = f"{INPUT_DIR}/ratings_2.txt"

HDFS_OUT  = "hdfs://localhost:9000/movie/output/bai1"
LOCAL_OUT = "/mnt/d/UIT 3rd year/BigData/ThucHanh/Lab3/output/bai1.txt"

# --- Bước 1: Đọc danh sách phim ---
def parse_movie(line):
    parts = line.split(",")
    return (int(parts[0]), parts[1])

id_to_title = sc.textFile(PATH_MOVIES).map(parse_movie).collectAsMap()

# --- Bước 2: Đọc ratings từ 2 file, map mỗi dòng → (movieId, (rating, 1)) ---
def parse_rating(line):
    cols = line.split(",")
    return (int(cols[1]), (float(cols[2]), 1))

all_ratings = sc.textFile(f"{PATH_RATE1},{PATH_RATE2}").map(parse_rating)

# --- Bước 3: Tính tổng rating và số lượt cho mỗi phim ---
def merge_counts(a, b):
    return (a[0] + b[0], a[1] + b[1])

aggregated = all_ratings.reduceByKey(merge_counts)

# --- Bước 4: Lọc phim >= 50 lượt, tính avg ---
def compute_avg(record):
    mid, (total, count) = record
    return (mid, round(total / count, 4), count)

qualified = aggregated.filter(lambda r: r[1][1] >= 50).map(compute_avg)

ranked = qualified.sortBy(lambda r: -r[1]).collect()

top_movie = ranked[0]
top_title = id_to_title.get(top_movie[0], "Unknown")

# --- In kết quả ---
header = f"  {'STT':<5} {'Tên phim':<50} {'Avg':>7} {'Count':>7}"
divider = "  " + "-" * 72
print(header)
print(divider)
for idx, (mid, avg, cnt) in enumerate(ranked, start=1):
    title = id_to_title.get(mid, "Unknown")
    print(f"  {idx:<5} {title:<50} {avg:>7.4f} {cnt:>7}")

print(f"\n>>> Phim có điểm TB cao nhất (>= 50 lượt đánh giá):")
print(f"    Tên phim  : {top_title}")
print(f"    MovieID   : {top_movie[0]}")
print(f"    Avg Rating: {top_movie[1]:.4f}")
print(f"    Số lượt   : {top_movie[2]}")

# --- Ghi file local ---
output_lines = [
    "\nDanh sách tất cả phim và điểm trung bình (>= 50 lượt):",
    header, divider
]
for idx, (mid, avg, cnt) in enumerate(ranked, start=1):
    title = id_to_title.get(mid, "Unknown")
    output_lines.append(f"  {idx:<5} {title:<50} {avg:>7.4f} {cnt:>7}")

output_lines += [
    f"\nPhim có điểm TB cao nhất (>= 50 lượt đánh giá):",
    f"  Tên phim  : {top_title}",
    f"  MovieID   : {top_movie[0]}",
    f"  Avg Rating: {top_movie[1]:.4f}",
    f"  Số lượt   : {top_movie[2]}",
]

# --- Lưu HDFS ---
sc.parallelize(ranked) \
  .map(lambda r: f"{id_to_title.get(r[0], 'Unknown')}::{r[0]}::{r[1]}::{r[2]}") \
  .saveAsTextFile(HDFS_OUT)
print(f"\n>>> Đã lưu kết quả lên HDFS: {HDFS_OUT}")

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)
with open(LOCAL_OUT, "w", encoding="utf-8") as fout:
    fout.write("\n".join(output_lines))
print(f">>> Đã lưu kết quả về local: {LOCAL_OUT}")
print("=" * 60 + "\n")

sc.stop()

from Threaded_Sparse_TFIDF.TFIDF import TF_IDF_Vectorizer
import time
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

def main():
	time_mat, memory_mat, k_mat = [], [], []
	for cross_validation in range(10):
		time_arr, memory_arr, k_arr = [], [], []
		for k in trange(1, 16, desc="CV-"+str(cross_validation)):
			start = time.time()
			tf_idf = TF_IDF_Vectorizer(use_cached=True, print_output=False)
			_, ranking, partitions = tf_idf.get_similarity_score("science fiction super hero movie", num_workers=k, return_worker_split=True)
			time_taken = time.time() - start
			time_arr.append(time_taken)
			mem = sys.getsizeof(partitions[0]['value'])/(1024**2)
			memory_arr.append(str(round(mem, 4)))
			k_arr.append(k)
		time_mat.append(time_arr)
		memory_mat.append(memory_arr)
		k_mat.append(k_arr)
	time_mat, memory_mat, k_mat = np.array(time_mat), np.array(memory_mat), np.array(k_mat)
	time_mat, memory_mat, k_mat = time_mat.astype(float), memory_mat.astype(float), k_mat.astype(int)
	time_mat = np.mean(time_mat, axis=0)
	memory_mat = np.mean(memory_mat, axis=0)
	k_mat = np.mean(k_mat, axis=0)
	df = pd.DataFrame({"num_workers":k_mat, "time":time_mat, "partition_size":memory_mat})
	df.to_csv("performance.csv", index=False)
	f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
	ax1.plot(k_mat, time_mat)
	ax1.set_title("Time-Taken vs Num-Workers")
	ax1.set_ylabel("time")
	ax2.plot(k_mat, memory_mat)
	ax2.set_title("Memory-Utilized vs Num-Workers")
	ax2.set_ylabel("memory")
	ax2.set_xlabel("num_workers")
	plt.savefig("performance.png")

if __name__ == '__main__':
	main()
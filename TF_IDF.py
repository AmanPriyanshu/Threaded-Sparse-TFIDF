import numpy as np
import os
from tqdm import tqdm
import re
from nltk.stem import PorterStemmer
import pickle
import threading

class TF_IDF_Vectorizer:
	def __init__(self, input_dir="./data/text/", output_dir="./output/", vocab=None, stemmer=None, use_cached=True):
		self.use_cached = use_cached
		self.input_dir = input_dir
		self.output_dir = output_dir
		self.file_array = os.listdir(self.input_dir)
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)
		if stemmer is None:
			stemmer = PorterStemmer()
		self.stemmer = stemmer
		if vocab is None:
			vocab = self.get_vocab()
		self.vocab = vocab
		self.idf = self.get_inverse_document_frequency()
		self.tf = self.get_term_frequency()
		self.similarities = []

	def get_vocab(self):
		if not os.path.exists(self.output_dir+"vocab.dat") or not self.use_cached:
			vocab = []
			bar = tqdm([self.input_dir+i for i in self.file_array])
			for file in bar:
				with open(file, "r", encoding="utf-8") as f:
					data = f.read()
				data = data.lower()
				data = re.sub("[^a-zA-Z]+", " ", data)
				words = [self.stemmer.stem(i) for i in data.split() if i!='']
				vocab.extend([i for i in words if len(i)>2 and len(i)<=8])
				vocab = list(set(vocab))
				bar.set_description("read - "+str(len(vocab)))
			bar.close()
			with open(self.output_dir+"vocab.dat", "w") as f:
				f.write("\n".join(vocab))
		else:
			print("Using cached... Vocab")
			with open(self.output_dir+"vocab.dat", "r") as f:
				vocab = [i.replace("\n", "") for i in f.readlines()]
		return vocab

	def get_term_frequency_document(self, data):
		documents = {"value": [], "columns": []}
		data = data.lower()
		data = re.sub("[^a-zA-Z]+", " ", data)
		words = [self.stemmer.stem(i) for i in data.split() if i!='']
		words = np.array([i for i in words if len(i)>2 and len(i)<=8])
		word_counts = np.unique(words, return_counts=True)
		words, counts = word_counts[0], word_counts[1]
		counter = 0
		for word,count in zip(words, counts):
			try:
				documents["columns"].append(self.vocab.index(word))
				documents["value"].append(count*self.idf[word])
				counter+=1
			except:
				continue
		return documents, counter

	def get_term_frequency(self):
		if not os.path.exists(self.output_dir+"tf.dat") or not self.use_cached:
			documents = {"value": [], "columns": [], "rows": [0], "files": []}
			bar = tqdm([self.input_dir+i for i in self.file_array], desc="extracting TF")
			for file in bar:
				with open(file, "r", encoding="utf-8") as f:
					data = f.read()
				document, counter = self.get_term_frequency_document(data)
				documents["value"].extend(document["value"])
				documents["columns"].extend(document["columns"])
				documents["rows"].append(counter+documents["rows"][-1])
				documents["files"].append(file)
			with open(self.output_dir+"tf.dat", "wb") as f:
				pickle.dump(documents, f)
		else:
			print("Using cached... TF")
			with open(self.output_dir+"tf.dat", "rb") as f:
				documents = pickle.load(f)
		return documents

	def get_inverse_document_frequency(self):
		if not os.path.exists(self.output_dir+"idf.dat") or not self.use_cached:
			df = {word:0 for word in self.vocab}
			bar = tqdm([self.input_dir+i for i in self.file_array], desc="extracting IDF")
			for file in bar:
				with open(file, "r", encoding="utf-8") as f:
					data = f.read()
					data = data.lower()
					data = re.sub("[^a-zA-Z]+", " ", data)
					words = [self.stemmer.stem(i) for i in data.split() if i!='']
					words = list(set([i for i in words if len(i)>2 and len(i)<=8]))
					for word in words:
						try:
							df[word] += 1
						except:
							pass
			N = len(self.file_array)
			idf = {key:np.log2(N/df[key]) for key in tqdm(df.keys(), disable=True)}
			with open(self.output_dir+"idf.dat", "wb") as f:
				pickle.dump(idf, f)
		else:
			print("Using cached... IDF")
			with open(self.output_dir+"idf.dat", "rb") as f:
				idf = pickle.load(f)
		return idf

	def get_document_similarity(self, document_vec, tf, name):
		similarity = []
		for idx in range(len(tf["rows"])-1):
			row_start = tf["rows"][idx]
			row_end = tf["rows"][idx+1]
			values = tf["value"][row_start:row_end]
			columns = tf["columns"][row_start:row_end]
			dot_product = 0
			for val,col in zip(values, columns):
				dot_product += val*document_vec[col]
			values = np.array(values)
			denom = np.sqrt(np.sum(document_vec**2))*np.sqrt(np.sum(values**2))
			if denom!=0:
				cosine = dot_product/(denom)
			else:
				cosine = 0
			similarity.append(cosine)
		self.similarities.append({"worker"+name: similarity})

	def get_similarity_score(self, document, top_k=5, num_workers=4):
		data = document
		data = data.lower()
		data = re.sub("[^a-zA-Z]+", " ", data)
		words = [self.stemmer.stem(i) for i in data.split() if i!='']
		words = np.array([i for i in words if len(i)>2 and len(i)<=8])
		word_counts = np.unique(words, return_counts=True)
		words, counts = word_counts[0], word_counts[1]
		document_vec = np.zeros(len(self.vocab))
		for word,count in zip(words, counts):
			try:
				document_vec[self.vocab.index(word)] = count*self.idf[word]
			except:
				pass
		worker_splits = []
		last_idx = None
		k = int(len(self.tf["rows"])/num_workers)
		for worker in range(num_workers):
			if worker==0:
				rows_arr = self.tf["rows"][0:(worker+1)*k]
			elif worker!=num_workers-1:
				rows_arr = self.tf["rows"][worker*k-1:(worker+1)*k]
			else:
				rows_arr = self.tf["rows"][worker*k-1:]
			rows_arr = np.array(rows_arr)
			columns_arr = self.tf["columns"][rows_arr[0]:rows_arr[-1]]
			values_arr = self.tf["value"][rows_arr[0]:rows_arr[-1]]
			tf = {"value": values_arr, "columns": columns_arr, "rows": rows_arr}
			worker_splits.append(tf)

		threads = []
		for worker in range(num_workers):
			threads.append(threading.Thread(target=self.get_document_similarity, args=(document_vec, worker_splits[worker],'t'+str(worker),), name='t'+str(worker)))
		self.similarities = []
		for worker in range(num_workers):
			threads[worker].start()
		for worker in range(num_workers):
			threads[worker].join()
		similarities = []
		for worker in range(num_workers):
			for items in self.similarities:
				name = list(items.keys())[0]
				if name=="workert"+str(worker):
					similarities.extend(items[name])
		similarities = np.array(similarities)
		ranking_indices = np.argsort(similarities)[::-1]
		ranking = np.array(self.tf["files"])
		ranking = ranking[ranking_indices]
		return similarities[:top_k], ranking[:top_k]

if __name__ == '__main__':
	tf_idf = TF_IDF_Vectorizer(use_cached=True)
	_, ranking = tf_idf.get_similarity_score("science fiction super hero movie")
	print(ranking)
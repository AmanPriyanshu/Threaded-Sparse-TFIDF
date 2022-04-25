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
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)
		if stemmer is None:
			stemmer = PorterStemmer()
		self.stemmer = stemmer
		if vocab is None:
			vocab = self.get_vocab()
		self.vocab = vocab
		self.tf = self.get_term_frequency()
		self.idf = self.get_inverse_document_frequency()
		self.similarities = []

	def get_vocab(self):
		if not os.path.exists(self.output_dir+"vocab.dat") or not self.use_cached:
			vocab = []
			bar = tqdm([self.input_dir+i for i in os.listdir(self.input_dir)])
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
				documents["value"].append(count)
				counter+=1
			except:
				continue
		return documents, counter

	def get_term_frequency(self):
		if not os.path.exists(self.output_dir+"tf.dat") or not self.use_cached:
			documents = {"value": [], "columns": [], "rows": [0]}
			bar = tqdm([self.input_dir+i for i in os.listdir(self.input_dir)])
			for file in bar:
				with open(file, "r", encoding="utf-8") as f:
					data = f.read()
				document, counter = self.get_term_frequency_document(data)
				documents["value"].extend(document["value"])
				documents["columns"].extend(document["columns"])
				documents["rows"].append(counter+documents["rows"][-1])
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
			bar = tqdm([self.input_dir+i for i in os.listdir(self.input_dir)])
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
			N = len(os.listdir(self.input_dir))
			idf = {key:np.log2(N/df[key]) for key in tqdm(df.keys())}
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
			cosine = dot_product/(np.sqrt(np.sum(document_vec**2))*np.sqrt(np.sum(values**2)))
			similarity.append(cosine)
		self.similarities.append({"worker"+name: similarity})

	def get_similarity_score(self, document, num_workers=4):
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
		for worker in range(num_workers):
			if worker!=num_workers-1:
				rows_arr = self.tf["rows"][worker*int(len(self.tf["rows"])/num_workers):(worker+1)*int(len(self.tf["rows"])/num_workers)]
			else:
				rows_arr = self.tf["rows"][worker*int(len(self.tf["rows"])/num_workers):]
			rows_arr = np.array(rows_arr)
			if last_idx is None:
				columns_arr = self.tf["columns"][rows_arr[0]:rows_arr[-1]]
				values_arr = self.tf["value"][rows_arr[0]:rows_arr[-1]]
				last_idx = rows_arr[-1]
			else:
				columns_arr = self.tf["columns"][last_idx:rows_arr[-1]]
				values_arr = self.tf["value"][last_idx:rows_arr[-1]]
				rows_arr = rows_arr - rows_arr[0]
			tf = {"value": values_arr, "columns": columns_arr, "rows": rows_arr}
			worker_splits.append(tf)

		threads = []
		for worker in range(num_workers):
			threads.append(threading.Thread(target=self.get_document_similarity, args=(document_vec, worker_splits[worker],'t'+str(worker),), name='t'+str(worker)))
		for worker in range(num_workers):
			threads[worker].start()
		for worker in range(num_workers):
			threads[worker].join()
		_ = [print(i.keys()) for i in self.similarities]
		similarities = []
		for worker in range(num_workers):
			for items in self.similarities:
				name = list(items.keys())[0]
				if name=="workert"+str(worker):
					similarities.extend(items[name])
		print(len(similarities))

if __name__ == '__main__':
	tf_idf = TF_IDF_Vectorizer()
	tf_idf.get_similarity_score("002989 1968 2009 movie Color Norman, Matt (I) War History Drama Thriller vietnam-war black-panther-party assassination digit-in-title based-on-film robert-f-kennedy mexico sequel black-panthers year-in-title riot second-part olympics usa black-power number-in-title 1968 martin-luther-king-jr. English Australia 120 Norman, Matt (I) Norman, Rebecca (II) Favelle, Michael Pederson, Dave Norman, Matt (I) Norman, Matt (I) Based on a true story from the film \"SALUTE\". 1968 is the follow up feature drama to Matt Norman's International award winning documentary feature \"SALUTE\". This film reminds us what it was to live through 1968 in Mexico and The United States. Racism, Poverty, and War were all part of this incredible year. A Civil war began in the streets of the United States when Martin Luther King Jr was killed. The year was full of turmoil and what followed was the making of a year that will be held in history as one of the most violent and horrifying in history around the World. This is")
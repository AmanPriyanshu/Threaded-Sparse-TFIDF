# Threaded-Sparse-TFIDF
Creating a repository for multithreading TF-IDF vectorization for similarity search using sparse matrices for computations.

[PyPi](https://pypi.org/project/Threaded-Sparse-TFIDF/)

## Usage:
```py
from TF_IDF import TF_IDF_Vectorizer

tf_idf = TF_IDF_Vectorizer(use_cached=True, print_output=False)
_, ranking = tf_idf.get_similarity_score("science fiction super hero movie", num_workers=k)
```

## Performance:
### Image:

![image](https://github.com/AmanPriyanshu/Threaded-Sparse-TFIDF/blob/main/performance.png)

### Table:
|num_workers|time              |partition_size     |
|-----------|------------------|-------------------|
|1.0        |1.1117637634277344|6.778499999999999  |
|2.0        |0.8195240020751953|3.4149000000000003 |
|3.0        |0.7357232332229614|2.2773             |
|4.0        |0.7232689380645752|1.7081             |
|5.0        |0.7375946760177612|1.3555999999999997 |
|6.0        |0.7682486534118652|1.1307000000000003 |
|7.0        |0.7640876531600952|0.9618             |
|8.0        |0.7513441801071167|0.8506             |
|9.0        |0.7795052766799927|0.7587             |
|10.0       |0.8141436100006103|0.6807             |
|11.0       |0.8003325223922729|0.6195000000000002 |
|12.0       |0.8441393852233887|0.5697             |
|13.0       |0.8490614175796509|0.5258000000000002 |
|14.0       |0.9322290658950806|0.48739999999999994|
|15.0       |0.8824400186538697|0.45729999999999993|

## Data
A subset of the **Information Retrieval Dataset - Internet Movie Database (IMDB)** specifically movies after the year 2007.

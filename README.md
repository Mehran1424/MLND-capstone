# MLND-capstone
My Udacity MLND capstone project code
This repository is a collection of Jupyter files I used for my capstone project in Udacity Machine Learning Nanodegree. The project was about performing sentiment analysis on 400k Amazon reviews. The raw data was obtained from [Kaggle](https://www.kaggle.com/bittlingmayer/amazonreviews).

Data required for this project to run can be found [here](https://drive.google.com/open?id=1nGZG-zMCy0xgp3aXLX_5xqy3QqI5AYuS).
Here's a description of the Jupyter files:
- **capstone_1_read_pickle:** Reads the training and test data and save it to pickle files
- **Capstone_removing_punctuation:** Removing punctuation from the comments to reduce the vocabulary size ('which,' and 'which' now count as only one word
- **capstone_2_sparse_matrix_generation:** creating a sparse matrix between comments are vocabulary space from pickle files using HashVectorizer class
- **capstone_2_sparse_matrix_generation-nopunc:** creating a sparse matrix between comments are vocabulary space from pickle files  with no punctuation
- **capstone_3_classification_sparse:** classification on the sparse matrix
- **capstone_classification_word2vec:** classification on the feature matrix of every comment, which is obtained by replacing every word with its corresponding word vector. Word vectors are obtained from Fasttext pretrained vectors
- **capstone_classification_word2vec-embed_lookup:** same thing only with using tf.nn.embedding_lookup method
- **capstone_CNN_1 to capstone_CNN_5:** using CNNs for classification (different files using different pretrained word vectors or different parameters
- **Glove_transformation:** transforming GloVe pretrained word vector files to a standard .npz file used in this project
- **Google_W2V:** same thing only for Google word vectors
- **word2vec_fasttext_conversion:** same thing for FastText word vectors
- **skipgram_tf-amazon_review_corpus:** Skipgram algorithm performed on the project corpus (Amazon reviews)
- **skipgram_tf:** Generic Skipgram algorithm implementation (from tensorflow examples)

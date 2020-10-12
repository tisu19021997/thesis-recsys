# Thesis' Recommender System
A REST-ful API that allows calls from clients (my React front-end) to perform tasks like building the models (Incremental SVD and Item-based KNN), recommendations for users, related products for products.

## Frameworks
* **Web Server**: `Flask` - https://flask.palletsprojects.com/en/1.1.x/
* **Machine Learning**:
    * Fast SVD: `funk-svd` - https://github.com/gbolmier/funk-svd
    * Collaborative Filtering: `surprise-scikit` - https://github.com/NicolasHug/Surprise
    * Data Analysis and Manipulation: `pandas` https://github.com/pandas-dev/pandas/
    * Utilities: `numpy` - https://github.com/numpy/numpy
* **Storage**: model files are stored on Microsoft Azure storage (student free tier).

## Core Folders and Files
* **algo/**: implementation of two core algorithms: Explicit Query Aspect Diversification (xQuAD) 
and the incremental version of Simon Funk's Singular Value Decomposition (SVD). 
Details about the methodology can be found in my other [repository][paper].
* **lib/funk_svd/**: the `funk-svd` library.
* **helper/**:
    * **accuracy.py**: some ranking metrics such as Average Recommendation Popularity (ARP),
    Average Popularity of Long-tail items (APLT), Average Coverage of Long-tail items (ACLT).
    * Others: some utils function, nothing important.
* **wrapper/RecSys.py**: the recommender system class. Built to be used on two algorithms: 
SVD for user recommendation and Item-based KNN for generating related products.


[paper]: https://github.com/tisu19021997/my-notebooks/tree/main/svd-xquad

 
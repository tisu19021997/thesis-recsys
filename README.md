# Thesis' Recommender System
* This repo contains only the source code of the recommender system which is a REST-ful API that allows calls from clients (my React front-end) to perform tasks like building the models (Incremental SVD and Item-based KNN), recommendations for users, related products for products.

## Live demo: https://thesis-frontend.herokuapp.com

:warning: **Note**: The live demo can be a bit slow as start up since `heroku` put applications in sleep mode after a short period of inactive.
Moreover, since the application is divided into three parts: ReactJS front-end, NodeJS back-end, Flask recommender system, each of them
only "wakes up" when its functions are being used (especially for recommender system, it only starts when log-in as admin 
and use the recommender system), some functions may be slow at first.

* :red_circle: **Login as admin**: username **"admin"**, password **"admin"** => head to `/recommender` route to use the recommender system (train/test/save model, dataset) and other small features.
* :large_blue_circle: **Login as usual user**: username **"A2GKMXRLI7KLFP"**, password **"123"** => explore the webiste functions like browsing/search/add-to-cart products, sort/filter products, etc.
   
## Paper I published along with this project: https://github.com/tisu19021997/my-notebooks/tree/main/svd-xquad

## Frameworks
* **Web Stack**: 
    * Recommender System: `Flask`
    * Front-end: `ReactJS`
    * Back-end: `NodeJs` and `ExpressJS`
    * Database: `MongoDB`
    
* **Machine Learning**:
    * Fast SVD: `funk-svd`
    * Collaborative Filtering: `surprise-scikit`
    * Data Analysis and Manipulation: `pandas`
    * Utilities: `numpy`
    
* **Storage**: model files are stored on **Microsoft Azure** storage (student free tier).

## Folders and Files
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

 
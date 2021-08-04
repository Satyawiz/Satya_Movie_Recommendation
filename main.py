from os import link
from flask import Flask,render_template,request
import joblib
import pandas as pd

movie_db=pd.read_csv("movies.csv")
movie_db.reset_index(level=0,inplace=True)

Movie_App=Flask(__name__)
vector=joblib.load("Vector.pkl")
kdtree=joblib.load("kdtree.pkl")
Cosine=joblib.load("Cosine.pkl")

@Movie_App.route('/')
def welcome():
    return render_template('Movie_recom.html')

@Movie_App.route('/getcategory',methods=['GET','POST'])
def getcategory():
    selectedval=request.get_data()
    selectedval=selectedval.decode('utf-8')
    embedded=vector.transform([selectedval]).toarray()
    distance, idx=kdtree.query(embedded.reshape(1,-1),k=10)
    option={}
    for i,value in list(enumerate(idx[0])):
        key='Option'+str(i+1)
        option[key]=movie_db['title'][value]
    print(option)
    return option

@Movie_App.route('/getlink',methods=['Get','POST'])
def getlink():
    selectedtitle=request.get_data()
    selectedtitle=selectedtitle.decode('utf-8')
    movie_index=movie_db[movie_db.title==selectedtitle]['index'].values[0]
    similar_index=list(enumerate(Cosine[movie_index]))
    sorted_similar_index=sorted(similar_index,key=lambda x:x[1],reverse=True)
    x=0
    links={}
    for movie in sorted_similar_index:
        key='Option'+str(x+1)
        links[key]=movie_db['poster_path'][movie[0]]
        print(movie_db['title'][movie[0]])
        print(movie_db['poster_path'][movie[0]])
        x+=1
        if x>4:
            break
    return links

if __name__=='__main__':
    Movie_App.run(debug=True)
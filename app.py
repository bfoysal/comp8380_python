from flask import Flask, request,jsonify
from flask_cors import CORS
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import os

app = Flask(__name__)
CORS(app)
sourcefile = '/mnt/3A0F13EC43ACDB83/workspace/dblp_outputs/dblpv10_titles'
outputdir = '/mnt/3A0F13EC43ACDB83/workspace/dblp_outputs/'
model = Word2Vec.load(outputdir+'w2v.model')
@app.route('/',methods=['GET'])
def get():
    return jsonify({'msg':"Hello World"})


@app.route('/similar/<word>',methods=['GET'])
def get_similar_word(word):
    words = word.split()
    similar_words =[]
    for w in words:
        s = model.wv.most_similar(w)[:5]
        similar_words.append([i[0] for i in s])
    print(similar_words)
    return jsonify(similar_words)
    


dataset=[]
def creat_dataset():
    with open(sourcefile,'r') as f:
        lines = f.readlines()
        for line in lines:
            dataset.append(line.split())
    print('dataset list length',len(dataset))

def create_model():
    with open(sourcefile,'r') as f:
        print('training model')
        model= Word2Vec(dataset,min_count=2,sg=1, workers=4)
        model.save(outputdir+'w2v.model')
        print('model saved')

# creat_dataset()
# create_model()




if __name__ == '__main__':
    app.run(debug=True)
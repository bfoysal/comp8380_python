from flask import Flask, request,jsonify
from flask_cors import CORS
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import os
import re
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)
sourcefile = '/mnt/3A0F13EC43ACDB83/workspace/dblp_outputs/dblpv10_titles'
outputdir = '/mnt/3A0F13EC43ACDB83/workspace/dblp_outputs/'
model = Word2Vec.load(outputdir+'w2v.model')
phrase_model = Word2Vec.load(outputdir+'phrase2vec.model')
@app.route('/',methods=['GET'])
def get():
    return jsonify({'msg':"Hello World"})


@app.route('/similar_words/<word>',methods=['GET'])
def get_similar_word(word):
    words = word.split()
    similar_words =[]
    for w in words:
        s = model.wv.most_similar(w)[:5]
        similar_words.append([i[0] for i in s])
    print(similar_words)
    return jsonify(similar_words)
    
@app.route('/similar_phrases/<phrase>',methods=['GET'])
def get_similar_phrase(phrase):
    similar_phrases = []
    p = phrase_model.wv.most_similar(phrase.strip().lower())[:5]
    similar_phrases.append([i[0] for i in p])
    print(similar_phrases)
    return jsonify(similar_phrases)



def creat_dataset():
    dataset=[]
    stop_words = set (stopwords.words('english'))
    with open(sourcefile,'r') as f:
        lines = f.readlines()
        # for line in lines:
        #     dataset.append(line.split())
        for line in lines:
            bigrams=[]
            tokens = re.split("[^a-zA-Z0-9]+", line.strip().lower())
            for i in range(len(tokens)-1):
                if not(tokens[i] in stop_words or tokens[i+1] in stop_words):
                    bigrams.append(tokens[i]+" "+tokens[i+1])
            # print(bigrams)
            dataset.append(bigrams)
            
    print('dataset list length',len(dataset))
    return dataset

def create_model():
    # with open(sourcefile,'r') as f:
        dataset = creat_dataset()
        print('training model')
        model= Word2Vec(dataset,min_count=2,sg=1, workers=4)
        model.save(outputdir+'phrase2vec.model')
        print('model saved')

# creat_dataset()
# create_model()

get_similar_phrase('software engineering')

if __name__ == '__main__':
    app.run(debug=True)
    
import pandas as pd
import matplotlib as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



def parse_data(file_path, venue):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.split("\t")
            data.append(dict(conference=venue,title=tokens[2].strip()))
    return data

def tokenize(title):
    tokeizer = RegexpTokenizer(r'[a-zA-Z]\w+')
    return tokeizer.tokenize(title)

def stemmer(words):
    stemmer = PorterStemmer()
    return " ".join(stemmer.stem(word) for word in words)

if __name__ == '__main__':
    source_path = "/mnt/3A0F13EC43ACDB83/workspace/dblp_outputs/"
    datasets =["icse","vldb"]
    data = []
    for dataset in datasets:
        data.append(parse_data(source_path+dataset+"_id.txt",dataset))

    icse = pd.DataFrame(data[0])
    vldb = pd.DataFrame(data[1])

    icse['class_target']=0
    vldb['class_target']=1
    df_list = [icse,vldb]
    df = pd.concat(df_list).reset_index(drop=True)

    df['tokens'] = df['title'].map(tokenize)
    df['stems'] = df['tokens'].map(stemmer)

    # print(df.sample(10))
    x = df['stems']
    y = df['class_target']

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size =0.25, random_state=5)

    # print(len(X_train), len(X_test))
    # print(len(Y_train), len(Y_test))
    
    mnnb_pipe = Pipeline(steps=[('tf',TfidfVectorizer()),('mnnb',MultinomialNB())])
    mnnb_grid = {
        'tf__max_features' :[2000,3000,5000],
        'tf__stop_words' :['english'],
        'tf__ngram_range':[(1,1),(1,2)],
        'tf__use_idf' :[True,False],
        'mnnb__alpha' :[0.1, 0.5, 1]
    }
    
    mnnb_gs = GridSearchCV(mnnb_pipe, mnnb_grid, n_jobs=-1)
    mnnb_gs.fit(X_train, Y_train)

    # print(mnnb_gs.score(X_train, Y_train))
    # print(mnnb_gs.score(X_test, Y_test))
    # print(mnnb_gs.best_params_) #best parameters
    
    mnnb_pred = mnnb_gs.predict(X_test)
    # df['predicts'] = mnnb_pred
    # print(df.sample(30))

    mnnb_conf = confusion_matrix(Y_test,mnnb_pred)
    # print("MNNB Cnf []: \n",mnnb_conf)
    mnnb_heatmap = sns.heatmap(mnnb_conf, cmap="YlGnBu", annot=True, square=True, fmt=".0f").get_figure()
    mnnb_heatmap.savefig('mnnb_heatmap.png',dpi=400)
    
    print(classification_report(Y_test,mnnb_pred))

    svc_pipe = Pipeline(steps=[('tf',TfidfVectorizer()),('svc',SVC())])
    svc_grid = {
        'tf__max_features' :[2000,3000,5000],
        'tf__stop_words' :['english'],
        'tf__ngram_range':[(1,1),(1,2)],
        'tf__use_idf' :[True,False],
        'svc__C' :[3,5],
        'svc__tol' :[1e-3,1e-5],
        'svc__class_weight':[{0:0.05,1:0.8},{0:0.07,1:1}]
    }
    
    svc_gs = GridSearchCV(svc_pipe, svc_grid, n_jobs=-1)
    svc_gs.fit(X_train, Y_train)
    # print("Best svc params:\n",svc_gs.best_params_)
    svc_pred = svc_gs.predict(X_test)
    svc_conf = confusion_matrix(Y_test,svc_pred)
    svc_heatmap = sns.heatmap(svc_conf, cmap="YlGnBu", annot=True, square=True, fmt=".0f").get_figure()
    svc_heatmap.savefig('svc_heatmap.png',dpi=400)
    
    print(classification_report(Y_test,svc_pred))

    # print(X_train.sample(20))
    



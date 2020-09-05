from flask import *
from flask import Flask, render_template, request
import os, string
import sys
import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)

@app.route('/skipgram_web_app',methods = ['POST'])
def hello():
    focus_word = request.form['focus_word']
    data_source = request.form['data_source']
    if data_source == 'Nexidia':
        new_model_nexidia = gensim.models.Word2Vec.load("/home/datascience/projects/sandeep/skipgram_web_app/hackathon_model")
        x = new_model_nexidia.wv.most_similar(positive=focus_word,topn=50)
    elif data_source == 'tNPS':
        tNPS_model = gensim.models.Word2Vec.load("/home/datascience/projects/sandeep/skipgram_web_app/tnps_model")
        x = tNPS_model.wv.most_similar(positive=focus_word,topn=50)
    elif data_source == 'XA':
        XA_model = gensim.models.Word2Vec.load("/home/datascience/projects/sandeep/skipgram_web_app/Hack_0811")
        x = XA_model.wv.most_similar(positive=focus_word,topn=50)
    elif data_source == 'IVR':
        IVR_model = gensim.models.Word2Vec.load("/home/datascience/projects/sandeep/skipgram_web_app/ivr_skipgram_model")
        x = IVR_model.wv.most_similar(positive=focus_word,topn=50)
    else:
        new_model_nexidia = gensim.models.Word2Vec.load("/home/datascience/projects/sandeep/skipgram_web_app/hackathon_model")
        x = new_model_nexidia.wv.most_similar(positive=focus_word,topn=50)
        
    x = str(x).strip('[]')
    x = x.replace(', (','<br/>')
    return render_template("result.html", result=x)

@app.route('/')
def index():
    return render_template('mainform.html')

if __name__ == '__main__':
   app.run(debug = True, host = '0.0.0.0', port = 6785)

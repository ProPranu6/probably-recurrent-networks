import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input, Dot, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.datasets import imdb
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
import wikipedia
import matplotlib.pyplot as plt
import pandas as pd
import copy
from math import *
%matplotlib inline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input, Dot, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy.special import softmax

import copy


class vecW2V():
    


  def __init__(self, modelName):
    self.model = None
    self.probLayer = None
    self.modelName = modelName
    self.modelInputDict = None
    self.modelVecSize = None
    self.modelCorpus = None
    self.modelVocab = None
    self.modelLinedVocab = None
    self.modelWordVectors = None
    self.freqDict = None
  
  def __call__(self, originals):
    result = []
    for org in originals:
      if org in self.modelVocab:
        one_hot_vector = enc.transform(np.array([org]).reshape(-1,1))
        result.append(np.dot(one_hot_vector, self.modelWordVectors))
      else :
        result.append(np.nan)
    return result

  def removeEle(self, li, ele):
    l = li
    try:
        while True:
            l.remove(ele)
    except ValueError:
        pass
    
    return l
    
  class helperFormat():
    
    def __init__(selfi):
      return
    
    def removeEle(selfi, li, ele):
      l = li
      try:
        while True:
            l.remove(ele)
      except ValueError:
        pass
    
      return l
    
    def removeStopWords(selfi, senList):
      
      noUseWords = stopwords.words('english')
      newSenList = []
    
      for lines in senList :
        
        words = copy.deepcopy(lines.split(" "))
        for nouse in noUseWords:
          if nouse in lines.split(" "):
           
            words = selfi.removeEle(words, nouse)
      
        newSenList.append(" ".join(words))
      
      return newSenList
  
  def uniquefy(self, seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
  
# Removing punctuations in string
# Using loop + punctuation string
#    for ele in string: 
#      if ele in punc: 
#        string = string.replace(ele, "")
  
#    return string

  def makeCorpus(self, title=None, fromText=None, allowPunctuation=False, allowLineBreaks=True):
    
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''

    if fromText == None:
      titlePage = wikipedia.page(title)
      corpus = (titlePage.content).lower()
  
    else:
      corpus = fromText.lower()

    if allowPunctuation == False:
      corpus = re.sub(r'==.*?==+', '', corpus)
      corpus = re.sub(r'[,;:\-\(\)\]\[\$%&\*_!<>@#"]','', corpus)
    
    corpus1 = re.sub(r'\n',' ', corpus)
    if allowLineBreaks == True:
      corpus2 = re.split(r'\.\s', corpus1)
    else:
      corpus2 = [corpus1]
    
    self.modelCorpus = corpus1

    return corpus1, corpus2


  def setFreqs(self):
    data = []
    freqDict = dict()
    for lis in self.modelLinedVocab.values():
      data += lis
    for vwds in self.modelVocab:
      freqDict[vwds] = data.count(vwds)
    
    self.freqDict = freqDict
    return "DONE"
    

    
  def generateSamplingDistribution(self, li, iterations=10):
    
    le = [self.modelVocab[x] for x in li]
    probDict = dict()
    totalFreqs = sum(list(self.freqDict.values()))
    for wds in le:
      probDict[wds] = (self.freqDict[wds])**0.75/(totalFreqs)**0.75
    
    pickUpList = []
    
    for wdi in li: 
      pickUpList += [wdi for x in range(ceil(probDict[self.modelVocab[wdi]]*iterations))]
    
    return pickUpList 


  def makeInputData(self, corpusSen, window, negativeSampling = 0.4, allowStopWords = False):
  
    global enc
    noUseWords = stopwords.words('english')
    togetherData = []

    vocab = []
    vocabLineWise = dict()
    lineCount = 0
    for lines in corpusSen :
      lines = re.sub("((\s+)\s)"," ", lines)
      lines = re.sub("^(\s+)|(\s+)$|","", lines)
      words = lines.split(" ")
      vocab += words
      vocabLineWise[lineCount] = words
      lineCount += 1
    
    vocab = self.uniquefy(vocab)
    if allowStopWords == False:
      for nouse in noUseWords:
        if nouse in vocab:
          vocab = self.removeEle(vocab, nouse)
    
    
    enc = OneHotEncoder(sparse=False)
    enc.fit(np.array(vocab).reshape(-1,1))

 
    for l in range(lineCount):
      temp = copy.deepcopy(vocabLineWise[l])
      for wds in vocabLineWise[l]:
      
        if wds not in vocab:
          
          temp = self.removeEle(temp, wds)
        
      vocabLineWise[l] = temp
      
    self.modelVocab = vocab
    self.modelLinedVocab = vocabLineWise
    _ = self.setFreqs()
  
    biggerOffSet = 0
    for sen in range(lineCount):
      xinputraw = vocabLineWise[sen]
    
      for c in range(len(xinputraw)):
        centreWord = xinputraw[c]

        contentIndices = range(max(0,c-window), min(len(xinputraw), c+window+1))
        negativecount = int(negativeSampling)
        candids = list(set(range(0, len(vocab))) - set(range(max(0+biggerOffSet,biggerOffSet + c-window), min(len(xinputraw)+biggerOffSet,biggerOffSet+c+window+1))))
        candids = self.generateSamplingDistribution(candids, iterations=window)

       
        if len(candids) == 0:
          noncontentIndices = [-1]
        else:
          noncontentIndices = random.choices(candids, k=negativecount)

        for pick1 in contentIndices :
          contextWord = xinputraw[pick1]
          togetherData.append((centreWord, contextWord, 1))
      
        for pick2 in noncontentIndices :
          if pick2 == -1 :
            continue
          noncontextWord = vocab[pick2]
          togetherData.append((centreWord, noncontextWord, 0))

        biggerOffSet += (len(xinputraw)-1)

    centreWordsAlone = []
    maybecontextWordsAlone = []
    similarity = []
    for dp in togetherData :
      centreWordsAlone.append(dp[0])
      maybecontextWordsAlone.append(dp[1])
      similarity.append(dp[2])
  
    xinput1 = enc.transform(np.array(centreWordsAlone).reshape(-1,1)).T
    xinput2 = enc.transform(np.array(maybecontextWordsAlone).reshape(-1,1)).T
    ylabel = np.array(similarity).reshape(1,-1)



    return xinput1, xinput2, ylabel, enc

  def makeModelAndInput(self, vecSize, corpus2, window=15, negativeSampling=5, allowStopWords=False, describeModel=True):
    
    corpusSen = corpus2
    xinput1, xinput2, ylabel, encoderObject= self.makeInputData(corpusSen, window=window, negativeSampling=negativeSampling, allowStopWords=allowStopWords)

    outputsize = ylabel.shape[0]
    xinput1 = xinput1.T
    xinput2 = xinput2.T
    ylabel = ylabel.T  #(None, 1)


    Xinp1 = Input(shape=(xinput1.shape[1],))
    Xinp2 = Input(shape=(xinput2.shape[1],))

    embed = Dense(vecSize, use_bias=False, name="embed")
    V1 = embed(Xinp1)
    V2 = embed(Xinp2)
    dot = Dot(axes=1)([V1, V2])
    dot = BatchNormalization()(dot)
    prob = Dense(1, activation='sigmoid', name="prob")(dot)
  
    model = Model(inputs=[Xinp1, Xinp2], outputs=prob)
    self.model = model
    self.probLayer = self.model.get_layer('prob')
    if describeModel == True:
      model.summary()
    
    inputDict = {"Input1":xinput1, "Input2":xinput2, "ylabel":ylabel}
    
    self.modelInputDict = inputDict
    self.modelVecSize = vecSize

    return model, inputDict

  def train(self, learning_rate=0.0009, metrics=["accuracy"], batch_size =32, epochs=100):
    model = self.model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=metrics)
    model.fit(x=[self.modelInputDict["Input1"], self.modelInputDict["Input2"]], y=self.modelInputDict["ylabel"], batch_size=batch_size, epochs=epochs, verbose=2)

    self.model = model
    self.probLayer = self.model.get_layer('prob')
    _ = self.describeWordVecs()

    return model
    
  def Sort_Tuple(self, tup): 
      
    # getting length of list of tuples
    lst = len(tup) 
    for i in range(0, lst): 
        for j in range(0, lst-i-1): 
            if (tup[j][1] < tup[j + 1][1]): 
                temp = tup[j] 
                tup[j]= tup[j + 1] 
                tup[j + 1]= temp 
    return tup 

  def sig(self, x):
    z = 1/(1 + np.exp(-x))
    return z

  def describeWordVecs(self):
    
    vectorizedWords = []
    for vectors in range(self.modelInputDict["Input1"].shape[1]):
      vectorizedWords.append(self.model.weights[0][vectors,:])
    
    vectorizedWords = np.array(vectorizedWords)

    self.modelWordVectors = vectorizedWords
    return vectorizedWords

  def autoFillList(self, word1=None, word2=None, topN=None,  printList=False):
    
    vectorizedWords = self.modelWordVectors
    originWord1 = enc.transform(np.array([word1]).reshape(-1,1))
    if word2 != None:
      originWord2 = enc.transform(np.array([word2]).reshape(-1,1))
      embedWord1 = np.dot(originWord1, vectorizedWords) 
      embedWord2 = np.dot(originWord2, vectorizedWords)
      
      matches = np.dot(embedWord1, embedWord2.T)
      den = np.linalg.norm(embedWord1)*np.linalg.norm(embedWord2)
      matches = matches/den
    
      reqProbs = matches
      if printList == True:
        print(*reqProbs[0])


      return reqProbs
    else:
      embedWord = np.dot(originWord1, vectorizedWords)
  
      matches = np.dot(tf.keras.utils.normalize(embedWord), tf.keras.utils.normalize(vectorizedWords).T)
  
      reqMatches = np.flip(np.argsort(matches))[0,0:topN]

      reqProbs = np.flip(np.sort(matches))[0,0:topN]

      originWord2 = np.zeros((self.modelInputDict["Input1"].shape[1],1))
      count = 0
      result = []
      for vindex in reqMatches :
        originWord2[vindex,0] = 1
    
        result.append((enc.inverse_transform(np.array([originWord2]).reshape(1,-1))[0,:], reqProbs[count]))
        if printList == True:
          print(enc.inverse_transform(np.array([originWord2]).reshape(1,-1)), reqProbs[count])
        count += 1
        originWord2[vindex,0] = 0


      return result

  def textCalc(self, printList=True, topN=10, **kwargs):
    vectorizedWords = self.modelWordVectors
    resultantVec = 0
    opCache = "+"
    vectorCount = 0
    if vectorizedWords.all() == None:
      return "No Vector Lookup Given"
    else :
      for keys, vals in kwargs.items():
      
        if "word" in keys:
          currentWord = enc.transform(np.array([vals]).reshape(-1,1))
          currentWord = np.dot(currentWord, vectorizedWords) 
          vectorCount +=1

          resultantVec = eval("resultantVec " + opCache + " currentWord")
        else :
          opCache = vals
    
      matches = np.dot(tf.keras.utils.normalize(resultantVec), tf.keras.utils.normalize(vectorizedWords).T)
  
      reqMatches = np.flip(np.argsort(matches))[0,0:topN]

      reqProbs = np.flip(np.sort(matches))[0,0:topN]

      originWord2 = np.zeros((self.modelInputDict["Input1"].shape[1],1))
      count = 0
      result = []
      for vindex in reqMatches :
        originWord2[vindex,0] = 1
    
        result.append((enc.inverse_transform(np.array([originWord2]).reshape(1,-1))[0,:], reqProbs[count]))
        if printList == True:
          print(enc.inverse_transform(np.array([originWord2]).reshape(1,-1)), reqProbs[count])
        count += 1
        originWord2[vindex,0] = 0


      return result, resultantVec

  def summarizeCorpus(self, summaryTagChoices = 3, summarizeEvery = 10, printTags = True):

    vectorizedWords = self.modelWordVectors
    filteredWds = self.modelVocab
    resultantVec = 0
    taggedContent = []
    contentNum = 0
    preWd = 0
    if vectorizedWords.all() == None:
      return "No Vector Lookup Given"
    else :
      for wds in range(len(filteredWds)):
      
        if filteredWds[wds] != "*" or filteredWds[wds] != "":
          currentWord = enc.transform(np.array([filteredWds[wds]]).reshape(-1,1))
          currentWord = np.dot(currentWord, vectorizedWords) 

          resultantVec += eval("currentWord")
        
        
        else :
          continue
      
        if (wds+1)%summarizeEvery == 0:

        
          matches = np.dot(tf.keras.utils.normalize(resultantVec/summarizeEvery), tf.keras.utils.normalize(vectorizedWords).T)
          reqMatches = np.flip(np.argsort(matches))[0,0:summaryTagChoices]
          reqProbs = np.flip(np.sort(matches))[0,0:summaryTagChoices]

          originWord2 = np.zeros((self.modelInputDict["Input1"].shape[1],1))
          count = 0
          result = []

          if printTags == True:
            print("Content Number : ", contentNum, "; The Orginial Content : ", " ".join(filteredWds[preWd:wds+1]), "\n")
          for vindex in reqMatches :
            originWord2[vindex,0] = 1
    
            result.append((enc.inverse_transform(np.array([originWord2]).reshape(1,-1))[0,:], reqProbs[count]))
            if printTags == True:
              print(enc.inverse_transform(np.array([originWord2]).reshape(1,-1)), reqProbs[count])
            count += 1
            originWord2[vindex,0] = 0
        
          if printTags == True:
            print("\n")
          contentNum += 1
          taggedContent.append(result)
          resultantVec = 0
          preWd = (wds+1)
      
      return taggedContent
  
  def predictText(self, word1=None, topN=None, printList = True):

    vectorizedWords = self.modelWordVectors
    originWord1 = enc.transform(np.array([word1]).reshape(-1,1))


    embedWord = np.dot(originWord1, vectorizedWords)

    scores =[]
    dots = np.dot(embedWord, vectorizedWords.T)
    
    for d in dots[0,:]:
      d = d.reshape(-1,1)
     
      scores.append(np.array(self.probLayer(d))[0,0])

    scores = np.array(scores)
    
    reqMatches = np.flip(np.argsort(scores))[0:topN]

    reqProbs = np.flip(np.sort(scores))[0:topN]

    originWord2 = np.zeros((self.modelInputDict["Input1"].shape[1],1))
    count = 0
    result = []
    for vindex in reqMatches :
      originWord2[vindex,0] = 1
    
      result.append((enc.inverse_transform(np.array([originWord2]).reshape(1,-1))[0,:], reqProbs[count]))
      if printList == True:
        print(enc.inverse_transform(np.array([originWord2]).reshape(1,-1)), reqProbs[count])
      count += 1
      originWord2[vindex,0] = 0


    return result



  def visualiseWordVec(self, targetDimensions=2, vecCount = None, model = None):

    filteredWds = self.modelVocab
    vectorizedWords = self.modelWordVectors
    labels = []
    tokens = []

    if model == None:

      for wds in range(len(filteredWds)):
      
          if (filteredWds[wds] != "*" or filteredWds[wds] != "" )and (filteredWds[wds] not in labels):
            currentWord = enc.transform(np.array([filteredWds[wds]]).reshape(-1,1))
            labels.append(filteredWds[wds])
            tokens.append(np.dot(currentWord, vectorizedWords)[0]) 
    else:
      
      for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    
    #print(tokens[0:2])
    Vtsned = TSNE(n_components=targetDimensions).fit_transform(tokens)

    x = []
    y = []
    for value in Vtsned:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16))
    if vecCount == None:
      num = len(x)
    else:
      num = vecCount 
    for i in range(num):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

    return





chat = open('/content/jaymama_chat.txt')
data = chat.read()
data = data.splitlines()
chat.close()

clean_data = []
pattern = "\[([^\[\]]+|(?0))*]"
for chat in data :
  rechat = re.sub("[\(\[].*?[\)\]]", "", chat) 
  if "\u200e" in rechat or "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them." in rechat or len(rechat.split(" "))<=1 or "https" in rechat:
    continue
  clean_data.append(rechat)
 
corpus_w2v = " ".join(clean_data)

chat_predict_w2v = vecW2V("chat_prediction")

_, corpus_sen_w2v = chat_predict_w2v.makeCorpus(fromText=corpus_w2v)

model_w2v, input_dict_w2v = chat_predict_w2v.makeModelAndInput(80, corpus_sen_w2v, window=5)

model_w2v = chat_predict_w2v.train()

from statistics import mean

def normalize_vecs(unnorm_vec):
  norm_vec = unnorm_vec/np.linalg.norm(unnorm_vec,keepdims=True)
  return norm_vec

def set_inputs_right(w2v_obj, seq_len="auto"):
  
  global X_in_word
  global Y_in_word
  
  lined_vocab_w2v = w2v_obj.modelLinedVocab

  corpus_rnn = []
  corpus_rnn_txt = []
  for lines in lined_vocab_w2v.values():
    result_txt = [line for line in lines]
    result = [normalize_vecs(line).reshape(-1).tolist() for line in w2v_obj(lines)]
    corpus_rnn.append(result)
    corpus_rnn_txt.append(result_txt)
  
  
  
  sen_lens = []
  for sen in corpus_rnn:
    sen_lens.append(len(sen))
  if seq_len == "auto":
    seq_len = int(mean(sen_lens)/3)
  

  X = []
  X_in_word = []
  Y = []
  Y_in_word = []

  for sen in corpus_rnn:
    #seq_len = 50
    num_records = len(sen) - seq_len
    #senc = sen.tolist()
    for i in range(num_records):
        X.append(sen[i:i+seq_len])
        Y.append(sen[i+seq_len])

  for sen in corpus_rnn_txt:
    #seq_len = 50
    num_records = len(sen) - seq_len
    #senc = sen.tolist()
    for i in range(num_records):
        X_in_word.append(sen[i:i+seq_len])
        Y_in_word.append(sen[i+seq_len])
        
  
  X = np.array(X)
  X_in_word = np.array(X_in_word)
  #X = np.expand_dims(X, axis=2)
  

  Y = np.array(Y)
  Y_in_word = np.array(Y_in_word)
  #Y = np.expand_dims(Y, axis=1)
  

  return X, Y, corpus_rnn, corpus_rnn_txt
  
def performance_by_similarity(true_vector, predicted_vector):
  score = np.dot(true_vector, predicted_vector.T)
  den = np.linalg.norm(true_vector)*np.linalg.norm(predicted_vector)
  score = score/den

  return score
def predict_next_words(w2v_obj, prediction_vector, sequence_index=-1, topN=10, printList=True):
  
  matches = [] #np.dot(w2v_obj.modelWordVectors, prediction_vector)
  for vec in range(w2v_obj.modelWordVectors.shape[0]):
    vector = w2v_obj.modelWordVectors[vec]
    matches.append(performance_by_similarity(vector, prediction_vector))
  
  reqMatches = np.flip(np.argsort(matches))[0:topN]

  reqProbs = np.flip(np.sort(matches))[0:topN]

  originWord2 = np.zeros((w2v_obj.modelInputDict["Input1"].shape[1],1))
  
  count = 0
  result = []
  result_in_vector = []
  if sequence_index  == -1:
    if printList == True:
      print("Out Of Input Prequel Words has following Predictions : \n")
    for vindex in reqMatches :
      originWord2[vindex,0] = 1
    
      result.append((enc.inverse_transform(np.array([originWord2]).reshape(1,-1))[0,:], reqProbs[count]))
      result_in_vector.append(w2v_obj.modelWordVectors[vindex])
      if printList == True:
        print(enc.inverse_transform(np.array([originWord2]).reshape(1,-1)), reqProbs[count])
      count += 1
      originWord2[vindex,0] = 0
  else:
    if printList == True:
      for wds in X_in_word[sequence_index]:#corpus_rnn_txt[sequence_index//divide_factor][sequence_index*seq_len :seq_len*(1+sequence_index)]:
        print(wds, end="-")
      print("\u0332".join(Y_in_word[sequence_index]+" "))

    for vindex in reqMatches :
        originWord2[vindex,0] = 1
      
        result.append((enc.inverse_transform(np.array([originWord2]).reshape(1,-1))[0,:], reqProbs[count]))
        result_in_vector.append(w2v_obj.modelWordVectors[vindex])
        if printList == True:
          print(enc.inverse_transform(np.array([originWord2]).reshape(1,-1)), reqProbs[count])
        count += 1
        originWord2[vindex,0] = 0


  return result, result_in_vector


def auto_code_c(seq_len, w2v_obj, rnn_model):
  print("Type in comments for a code snippet with minimum", seq_len," words length :\n")
  typed = input("")
  word_list = typed.split(" ")
  auto_predict_upto = int(input("Enter upto how many words the code should be generated : "))
  for wds in typed.split(" "):
    if wds not in w2v_obj.modelVocab:
      while (wds in word_list):
        word_list.remove(wds)
  typed = " ".join(word_list)

  if len(word_list) <seq_len:
    print("Insufficent vocab words in typed content")

  else:
    typed_txt = typed.split(" ")
    typed = [normalize_vecs(wd) for wd in w2v_obj(typed.split(" "))]

    X_custom = []
    X_custom_in_word = []
    num_records = len(typed) - seq_len +1
      
    for i in range(num_records):
        X_custom.append(typed[i:i+seq_len])
        X_custom_in_word.append(typed_txt[i:i+seq_len])

    m  = rnn_model.input_shape[-1]
    X_custom = np.array(X_custom).reshape(-1, seq_len, m)
    
    X_custom_cache = [] + list(X_custom.reshape(-1, m)[:seq_len])
  
    pred_cache = []
    print("Generated code snippet : \n")
    if auto_predict_upto == -1:
      pred= "start"
      i = 0
      while (pred != "#end" and pred != "#python"):
        consider = np.array(X_custom_cache[len(X_custom_cache)-seq_len:])
    
        pred, _ = predict_next_words(w2v_obj, rnn_model.predict(np.expand_dims(consider, axis=0))[0], topN=1, printList=False)
        pred_vector = w2v_obj(pred[0][0].tolist())[0][0]
        pred_vector = normalize_vecs(pred_vector)
        pred = pred[0][0][0]

        X_custom_cache.append(pred_vector)
        pred_cache.append(pred)
        #pred_cache.pop(0)
        X_custom_cache.pop(0)
        if (i%10 == 0):
          print("\n")
        print("", end=" ")
        print(pred, end=" ")
        i+=1

    else:
      for i in range(auto_predict_upto):
        consider = np.array(X_custom_cache[len(X_custom_cache)-seq_len:])
      
        pred, _ = predict_next_words(w2v_obj, rnn_model.predict(np.expand_dims(consider, axis=0))[0], topN=1, printList=False)
        pred_vector = w2v_obj(pred[0][0].tolist())[0][0]
        pred_vector = normalize_vecs(pred_vector)
        pred = pred[0][0][0]

        X_custom_cache.append(pred_vector)
        pred_cache.append(pred)
        #pred_cache.pop(0)
        X_custom_cache.pop(0)
        if (i%10 == 0):
          print("\n")
        print("", end=" ")
        print(pred, end=" ")

X, Y, corpus_rnn, corpus_rnn_txt = set_inputs_right(chat_predict_w2v, seq_len=4)
X.shape, Y.shape

import recurrent_neural_networks
from recurrent_neural_networks import rnns
models = rnns()
model = models.make_many_one_model(serial_input_card=4,architecture_per_ss=[('inpk',80),('jnk', 160),('ffk', 40),('ffk', 80)], activation="linear", hidden_activations="relu")

model.compile(optimizer=Adam(learning_rate=0.002), loss="mse", metrics=['mae'])
model.fit(X, Y, epochs=200)

scores = []
pred_vs = model.predict(np.expand_dims(X, axis=0).reshape(-1, 4, 80)) 
threshold = 0.7
for sam in range(X.shape[0]):
  pred_v = pred_vs[sam]
  score = performance_by_similarity(Y[sam], pred_v)
  scores.append(score)

best_preds_seq_nums = []
count =0
for sc in scores:
  if sc >=threshold:
    best_preds_seq_nums.append(count)
  count +=1

print(best_preds_seq_nums)
print(len(best_preds_seq_nums)/len(scores)*100, "% samples produced predicted words with",threshold*100, "% match with the original word")
print("NOTE : 60% match between original and predicted word would be enough to rightly predict the word")


text =" ".join(Y_in_word[0:500])
#text = Y_in_word[0:500]
print("Original Text upto ", 500,"words :\n", text)

gen_text = []
for wds in range(0, 500):
  next_wds = predict_next_words(chat_predict_w2v, wds,model.predict(np.expand_dims(X[wds], axis=0))[0], printList=False, topN=1)[0][0]
  
  gen_text += list(next_wds)

gen_text = " ".join(gen_text)

print("AI Generated Text upto", 500,"words :\n", gen_text)
"""
print("Matched Original And Generated Words\n")
count = 1
for wds in range(0,500):
  if text[wds] == gen_text[wds]:
    print(count, ":", text[wds])
    count +=1
"""
a = "end"

typed = input("Type in content with minimum 4 words length :\n")
seq_len = 4
word_list = typed.split(" ")
for wds in typed.split(" "):
  if wds not in chat_predict_w2v.modelVocab:
    while (wds in word_list):
      word_list.remove(wds)
typed = " ".join(word_list)

if len(typed) <4:
  raise "Insufficent vocab words in typed content"

typed_txt = typed.split(" ")
typed = [normalize_vecs(wd) for wd in chat_predict_w2v(typed.split(" "))]

X_custom = []
X_custom_in_word = []
num_records = len(typed) - seq_len +1
  
for i in range(num_records):
    X_custom.append(typed[i:i+seq_len])
    X_custom_in_word.append(typed_txt[i:i+seq_len])


X_custom = np.array(X_custom).reshape(-1, 4, 80)


print("predictions : \n")
for index in range(X_custom.shape[0]):
  pred = predict_next_words(chat_predict_w2v, model.predict(np.expand_dims(X_custom[index], axis=0))[0], topN=3, printList=False)
  
  print(" ".join(X_custom_in_word[index]), ":", pred)

  


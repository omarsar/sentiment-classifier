__author__ = 'ellfae'
#############################################################################################
#This Sentiment Classifier was built from scratch
#and if you would like the complete project together with labeled dataset
#feel free to contact me here: ellfae@gmail.com
#Here you will find some useful functions that you can reuse in your NLP projects.
#Documentation coming soon
#############################################################################################


from nltk.corpus import stopwords
#from __future__ import division
import re
import collections, itertools
import csv
import sys
import os
import codecs
import json
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from random import shuffle
from nltk.tokenize import WhitespaceTokenizer
import nltk.classify
from sklearn.svm import LinearSVC
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import operator
import nltk.classify.util, nltk.metrics
import matplotlib.pyplot as plt
import  matplotlib.animation as animation
from nltk.probability import FreqDist, ConditionalFreqDist


#Two global variables that keep track of negative and positive tweets.
globalnegative = 0
globalpositive = 0


class featureReduction:

    #Initializing function to write the necessary files.
    def __init__(self):
        #CSV files to keep tweets
        p = open('postweets.csv','wb')
        self.postweets = csv.writer(p,delimiter = ',')
        ne = open('negtweets.csv','wb')
        self.negtweets = csv.writer(ne,delimiter = ',')
        i = open('inftweets.csv','wb')
        self.inftweets = csv.writer(i,delimiter = ',')
        #extra = open('extratweets66.csv','wb')
        #self.extra = csv.writer(extra,delimiter = ',')

        self.cachedStopWords = stopwords.words("english")
        #punctuation = re.compile(r'[.?!,":;\n\s]') #this regular expression removes the punctuation marks from the words.
        #self.punctuation = re.compile(r'[-.?!,":;()@*/<|0-9]')
        self.punctuation = re.compile(r'[.?!,"|0-9]')
        self.amountneg = 0
        self.amountpos = 0
        self.amountinf = 0
        self.amountwasted = 0

        #self.st = LancasterStemmer()
        self.st = WordNetLemmatizer()

        for n in range(1,11):
            f = str(n)
            try:
                corpus = csv.reader(open("labeled/data"+f+".csv",'rb'), delimiter = ',')
                for row in corpus:

                    final = self.tweet_reduction(row[0])# this takes care of the stemming and tweet reduction phase

                    average = self.calc_average(row[1],row[2],row[3])

                    self.write_results(row[0],final,row[1],row[2],row[3],average[0],average[1],average[2])

            except:
                print "File not found!"

        print "Amount Negative:",self.amountneg
        print "Amount Positive:",self.amountpos
        print "Amount Informal:",self.amountinf

        ne.close()
        p.close()
        i.close()


    #######################
    #FUNCTIONS
    #######################
    #Determines annotating average
    def calc_average (self,one,two,three):

        if (str(one) == str(two) and str(one) != str(three)):
            result1 = 66
            result2 = str(one)
            resultodd = str(three)
        elif (str(one) == str(two) and str(one) ==  str(three)):
            result1 = 100
            result2 = str(one)
            resultodd = str(one)
        elif (str(one) != str(two) and str(one) ==  str(three)):
            result1 = 66
            result2 = str(one)
            resultodd = str(two)
        elif (str(one) != str(two) and str(two) ==  str(three)):
            result1 = 66
            result2 = str(two)
            resultodd = str(one)
        elif (str(one) != str(two) and str(two) !=  str(three)):
            result1 = 0
            result2 = "none"
            resultodd = "none"
        else:
            print "found more!"

        #print result1,result2
        return  (result2,result1,resultodd)

    #main function to simplify tweets into its normal form
    def tweet_reduction(self,tweet):
        REMOVE_LIST = ["google","apple","microsoft","twitter"]
        remove = '|'.join(REMOVE_LIST)
        regex = re.compile(r'\b('+remove+r')\b',flags=re.IGNORECASE)

        stepone = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',tweet)# replace urls
        #steptwo = re.sub(r'#(\w)+','',stepone) #replace the hashtags with HT
        steptwo = re.sub(r'#(\w+)',r'\1',stepone)
        #print "Oroginal:" , stepone
        #print "With hash: ",steptwo
        stepthree =  re.sub(r'@(\w)+','',steptwo) #replce the usernames or mentions
        stepfour = re.sub(r'((\w)\2{2,})',r'\2',stepthree) #replace all words repeated more than two times
        stepfive = regex.sub("",stepfour)
        lowercase = stepfour.lower()

        final = ' '.join([word for word in lowercase.split() if word not in self.cachedStopWords] )#checks for stop words and also get specifies length of word that it agrees on
        word_list = [self.punctuation.sub(" ", word) for word in final]
        final2 = ''.join(word_list)
        final2 = re.sub(' +',' ',final2) # this should remove all those extra whitespaces
        final2 = final2.strip() # strip extra whitespaces in front of tweet

        results = []
        for sword in final2:
            sword = self.st.lemmatize(sword).replace("[","").replace("]","")
            results.append(sword)

        preresult = ''.join(results)

        postresult = ' '.join([w for w in preresult.split() if len(w)>=2])

        postresult2 = self.getPOS(postresult)
        #print "Original", (postresult)
        #print "After Stemming",(postresult2)
        return ''.join(postresult2)

    #In charge of returning the Part of Speech tag obtained from the words.
    def getPOS(self,w):
        stemmedcomplete = []
        wordnet_tag = {'NN':'n','JJ':'a','VB':'v','RB':'r'}
        tokens  = WhitespaceTokenizer().tokenize(w)
        tagged = nltk.pos_tag(tokens)
        for t in tagged:
            #print t[0],t[1][:1]
            try:
                result = self.st.lemmatize(t[0],wordnet_tag[t[1][:2]])
            except:
                result = self.st.lemmatize(t[0])
            stemmedcomplete.append(result)

        return ' '.join(stemmedcomplete)

    #function that collects and writes the tweets that meet the criteria to its specific file
    def write_results(self,original,finaltweet,one, two,three, label, average,odd):
        if (label == 'p' and average >=100):
            self.postweets.writerow([str(finaltweet),str("positive")])
            #global amountpos
            self.amountpos+=1
        elif (label == 'n' and average >=100):
            self.negtweets.writerow([str(finaltweet),str("negative")])
            #global amountneg
            self.amountneg+=1
        elif (label == 'i' and average >=100):
            self.inftweets.writerow([str(finaltweet),str("neutral")])
            self.amountinf+=1
       # elif ((label == 'i' or label == 'n' or label == 'p') and average == 66):
        #    self.extra.writerow([str(original),str(label)])
        else:
            self.amountwasted+=1


#Second class with Classifier code:
class nbClassifier:
    def __init__(self):
        #self.lmtzr = WordNetLemmatizer()
        self.sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle')
        self.st = LancasterStemmer()
        self.test_tweets = []

        #Read the tweets
        self.inf1 = csv.reader(open("inftweets.csv",'rb'), delimiter = ',')
        self.pos = csv.reader(open("postweets.csv",'rb'), delimiter = ',')
        self.neg = csv.reader(open("negtweets.csv",'rb'), delimiter = ',')
        file = open('test.txt','wb')
        bag = csv.writer(open("bagofwords.csv",'wb'), delimiter = ',')

        inftweets = self.getTweets(self.inf1)
        postweets = self.getTweets(self.pos)
        negtweets = self.getTweets(self.neg)

        #Used to normalize the distribution of tweets
        tweetsizes = []

        self.infwords = []
        self.poswords = []
        self.negwords = []

        tweetsizes.append(inftweets.__len__())
        tweetsizes.append(postweets.__len__())
        tweetsizes.append(negtweets.__len__())

        #shuffle(inftweets)
        #shuffle(postweets)
        #shuffle(negtweets)

        #This chops the tweets to the minimum of all the tweets collected.
        inftweets = self.cutList(inftweets,min(tweetsizes))
        postweets = self.cutList(postweets,min(tweetsizes))
        negtweets = self.cutList(negtweets,min(tweetsizes))

        #try to remove the duplicates in each set


        tweets = self.combineTweets(inftweets,postweets,negtweets)
        #shuffle(tweets)
        #print tweets
        pos_tweets = tweets
        #print pos_tweets
        #pos_tweets = self.ChiSquareAnalysis(tweets)

        self.word_features = self.getFeatures2(self.getWords(pos_tweets))

        self.myClassifier(pos_tweets,self.test_tweets)

    #######################
    ##Begin of my functions
    #######################

    def getFeatures2 (self,wordlist):
        #word_fd  = nltk.FreqDist()
        word_fd = FreqDist()
        label_word_fd = ConditionalFreqDist()

        
        for word in self.infwords:
            word_fd[word.lower()]+=1
            label_word_fd['inf'][word.lower()]+=1
            #word_fd.inc(word.lower())
            #label_word_fd['inf'].inc(word.lower())
        for word in self.negwords:
            word_fd[word.lower()]+=1
            label_word_fd['neg'][word.lower()]+=1
            #word_fd.inc(word.lower())
            #label_word_fd['neg'].inc(word.lower())
        for word in self.poswords:
            word_fd[word.lower()]+=1
            label_word_fd['pos'][word.lower()]+=1
            #word_fd.inc(word.lower())
            #label_word_fd['pos'].inc(word.lower())

        
        pos_word_count = label_word_fd['pos'].N()
        neg_word_count = label_word_fd['neg'].N()
        inf_word_count = label_word_fd['inf'].N()
        total_word_count = pos_word_count + neg_word_count +inf_word_count

        word_scores = {}

        for word,freq  in word_fd.iteritems():
            pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],(freq,pos_word_count),total_word_count)
            neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],(freq,neg_word_count),total_word_count)
            inf_score = BigramAssocMeasures.chi_sq(label_word_fd['inf'][word],(freq,inf_word_count),total_word_count)
            word_scores[word] = pos_score+neg_score+inf_score

        best = sorted(word_scores.iteritems(),key=lambda (w,s): s,reverse=True)[:5000]
        bestwords =set ([w for w, s in best])

        self.word_features = bestwords
        #print self.word_features
        #print "The end of getFeature2"
        return self.word_features

    def getFeatures(self,wordlist):
        wordlist = nltk.FreqDist(wordlist)
        #print wordlist
        self.word_features = wordlist.keys()
        #print self.word_features

        return self.word_features

    def ChiSquareAnalysis2(self,t):
        chi = open("chiresults.txt",'wb')
        label,tweet = [],[]
        for tw, se in t:
            label.append(se)
            tweet.append(' '.join(tw))
            #print "TWeet array:",tweet
        label_idx = map(lambda x: label.index(x), label)
        #print  label
        #print label_idx
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(min_df=1,ngram_range=(1,2))
        x=vectorizer.fit_transform(tweet)


        from sklearn.feature_selection import chi2

        chi2score = chi2(x,label_idx)[0]
        import matplotlib.pyplot as plt

        plt.figure(figsize=(25,25))
        wscores = zip(vectorizer.get_feature_names(),chi2score)
        wchi2 = sorted(wscores,key=lambda x:x[1])
        #print wchi2
        #chi.write(str(wchi2))
        topchi2 = zip(*wchi2[-200:]) #This is a list of the terms and their assigned values.
        #chi.write(str(topchi2))
        #topchi2 = zip(*wchi2)
        x = range(len(topchi2[1])) #this is to represent the values/numbers
        labels = topchi2[0] #this will contatin the labels
        plt.barh(x,topchi2[1], align='center', alpha=0.2, color='g')
        plt.plot(topchi2[1],x,'-o',markersize=5,alpha=0.8, color='g')
        plt.yticks(x, labels)
        plt.xlabel('$\chi^2$')
        #plt.show()
        #plt.savefig("test.png")
        from collections import  Counter
        from collections import defaultdict

        words = topchi2[0] #also contains the words.

        label_words = defaultdict(lambda: Counter(),{})

        for word in words:
            for (i,st) in enumerate(label):
                m=re.search('(^|\W)'+word+'($|\W)',tweet[i])

                if not m: continue
                label_words[word].update({st:1})

        for w in sorted(label_words.keys()):
            if sum(label_words[w].values())<60: continue
            common = label_words[w].most_common(3)
            print "Word:",w
            labelarray = []
            valuearray = []
            indicatorlabel = "none"
            for l, n in common:
                labelarray.append(l)
                valuearray.append(n)
            print labelarray
            print valuearray
            if len(labelarray)>=2:
                if (valuearray[0] - valuearray[1]) >=10:
                    indicatorlabel = labelarray[0]

            #print indicatorlabel
            for tw,se in t:
                if se == "neutral" and (indicatorlabel != "neutral" or indicatorlabel!="none"):
                    if w in tw:
                        #print w,tw,se
                        tw.remove(w)
                        #print "After removal",tw
                #this will remove neutral words from negative and postive twets
                elif indicatorlabel == "neutral" and (se == "positive" or se == "negative"):
                    if w in tw:
                        tw.remove(w)

        for tw2,se2 in t:
            #print tw2
            if se2 == "neutral":
                if re.search('don\'t',' '.join(tw2)): print tw2, se2

        print "REached the end!"
        return t

    def ChiSquareAnalysis(self,t):
        label,tweet = [],[]
        for tw, se in t:
            label.append(se)
            tweet.append(' '.join(tw))
        label_idx = map(lambda x: label.index(x), label)
        #print  label
        #print label_idx
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(min_df=1)
        x=vectorizer.fit_transform(tweet)

        from sklearn.feature_selection import chi2

        chi2score = chi2(x,label_idx)[0]
        import matplotlib.pyplot as plt

        plt.figure(figsize=(25,25))
        wscores = zip(vectorizer.get_feature_names(),chi2score)
        wchi2 = sorted(wscores,key=lambda x:x[1])
        topchi2 = zip(*wchi2[-200:])
        x = range(len(topchi2[1]))
        labels = topchi2[0]
        plt.barh(x,topchi2[1], align='center', alpha=0.2, color='g')
        plt.plot(topchi2[1],x,'-o',markersize=5,alpha=0.8, color='g')
        plt.yticks(x, labels)
        plt.xlabel('$\chi^2$')
        #plt.show()
        #plt.savefig("test.png")
        from collections import  Counter
        from collections import defaultdict

        words = topchi2[0]

        label_words = defaultdict(lambda: Counter(),{})

        for word in words:
            for (i,st) in enumerate(label):
                m=re.search('(^|\W)'+word+'($|\W)',tweet[i])
                if not m: continue
                label_words[word].update({st:1})

        for w in sorted(label_words.keys()):
            if sum(label_words[w].values())<13: continue
            common = label_words[w].most_common(2)
            #print w
            labelarray = []
            valuearray = []
            indicatorlabel = "none"
            for l, n in common:
                labelarray.append(l)
                valuearray.append(n)
            print labelarray
            print valuearray
            if len(labelarray)>=2:
                if (max(valuearray) - min(valuearray)) >=10:
                    indicatorlabel = labelarray[0]

            #print indicatorlabel
            for tw,se in t:
                if indicatorlabel == se:
                    continue
                elif indicatorlabel == "none":
                    continue
                else:
                    if w in tw:
                        tw.remove(w)
                    else:
                        continue

        print "REached the end!"
        return t

    def getTweets(self,file):
        inftweets = []
        for row in file:
            #an array consisting of parsed tweet, original tweet and sentiment
            inftweets.append((row[0],row[1]))
        return inftweets

    def checkIfAdjective (self,word):
        tokens  = WhitespaceTokenizer().tokenize(word)
        tagged = nltk.pos_tag(tokens)
       # namedEnt = nltk.ne_chunk(tagged, binary=True)
        return tagged

    #the patterns that are called for extraction
    def checkIfAdjectivev2 (self,word):
        wordsfound=[]
        tokens  = WhitespaceTokenizer().tokenize(word)
        tagged = nltk.pos_tag(tokens)
        namedEnt = nltk.ne_chunk(tagged, binary=True)
        pattern1 = re.findall(r'(\w+\'*\w+)/JJ\s*(\w+)/NN',str(namedEnt))
        pattern2 = re.findall(r'(\w+\'*\w+)/JJ\s*(\w+)/NNS',str(namedEnt))

        pattern3 = re.findall(r'(\w+\'*\w+)/RB\s*(\w+)/JJ',str(namedEnt))
        pattern4 = re.findall(r'(\w+\'*\w+)/RBR\s*(\w+)/JJ',str(namedEnt))
        pattern5 = re.findall(r'(\w+\'*\w+)/RBS\s*(\w+)/JJ',str(namedEnt))

        pattern6 = re.findall(r'(\w+\'*\w+)/JJ\s*(\w+)/JJ',str(namedEnt))

        pattern7 = re.findall(r'(\w+\'*\w+)/NN\s*(\w+)/JJ',str(namedEnt))
        pattern8 = re.findall(r'(\w+\'*\w+)/NNS\s*(\w+)/JJ',str(namedEnt))

        pattern9 = re.findall(r'(\w+\'*\w+)/RB\s*(\w+)/VB',str(namedEnt))
        pattern10 = re.findall(r'(\w+\'*\w+)/RB\s*(\w+)/VBD',str(namedEnt))
        pattern11 = re.findall(r'(\w+\'*\w+)/RB\s*(\w+)/VBN',str(namedEnt))
        pattern12 = re.findall(r'(\w+\'*\w+)/RB\s*(\w+)/VBG',str(namedEnt))

        pattern13 = re.findall(r'(\w+\'*\w+)/RBR\s*(\w+)/VB',str(namedEnt))
        pattern14 = re.findall(r'(\w+\'*\w+)/RBR\s*(\w+)/VBD',str(namedEnt))
        pattern15 = re.findall(r'(\w+\'*\w+)/RBR\s*(\w+)/VBN',str(namedEnt))
        pattern16 = re.findall(r'(\w+\'*\w+)/RBR\s*(\w+)/VBG',str(namedEnt))

        pattern17 = re.findall(r'(\w+\'*\w+)/RBS\s*(\w+)/VB',str(namedEnt))
        pattern18 = re.findall(r'(\w+\'*\w+)/RBS\s*(\w+)/VBD',str(namedEnt))
        pattern19 = re.findall(r'(\w+\'*\w+)/RBS\s*(\w+)/VBN',str(namedEnt))
        pattern20 = re.findall(r'(\w+\'*\w+)/RBS\s*(\w+)/VBG',str(namedEnt))

        for w1,w2 in pattern1+pattern2+pattern3+pattern4+pattern5+pattern6+pattern7+pattern8+pattern9+pattern10+pattern11+pattern12+pattern13+\
            pattern14+pattern15+pattern16+pattern17+pattern18+pattern19+pattern20:
            if w1 and w2:
                #print w1,w2
                doubles = w1+" "+w2
                #print doubles
                if w1 in wordsfound:
                    pass
                else:
                    wordsfound.append(w1)
                if w2 in wordsfound:
                    pass
                else:
                    wordsfound.append(w2)
                #This adds the bigrams ...This will be excluded for now.
                if doubles in wordsfound:
                    pass
                else:
                    wordsfound.append(doubles)

        return wordsfound

    #combine tweets into their pertaining files
    def combineTweets(self,inf,pos,neg):
        alltweets = []
        randomtracker = 0

        bag = csv.writer(open("bagofwords.csv",'wb'), delimiter = ',')

        for (words,sentiment) in inf+ pos+neg:
            #filterwords = [e.replace("'","").replace("[","").replace("]","") for e in words.split() if len(e)>=3]
            filterwords = [e for e in words.split()]
            randomtracker +=1
            if ((randomtracker %10) == 0 ): #This is for selecting the type of cross validation
                #global  test_tweets
                self.test_tweets.append((filterwords,sentiment))
            else:
                alltweets.append((filterwords,sentiment))

        return alltweets

    #this is to get the Bigrams for each document(tweet) no used here
    def getBigrams(self,tweet):
        joinedtweet = tweet #' '.join(tweet)
        tosendtweet = []
        for item in nltk.bigrams (joinedtweet.split()): tosendtweet.append(' '.join(item))
        print tosendtweet
        return tosendtweet

    #to get the trigrams
    def getTrigrams(self,tweet):
        joinedtweet = ' '.join(tweet)
        tosendtweet = []
        for item in nltk.trigrams(joinedtweet.split()): tosendtweet.append(' '.join(item))

        return tosendtweet


    def writeBagOfWords(self,inf, neg, pos):
        p = open('poswords.csv','wb')
        post = csv.writer(p,delimiter = ',')
        n = open('negwords.csv','wb')
        negt = csv.writer(n,delimiter = ',')
        i = open('infwords.csv','wb')
        inft = csv.writer(i,delimiter = ',')

        #sorting
        dictionaryListInf = sorted(inf.iteritems(),key=operator.itemgetter(1))
        dictionaryListPos = sorted(pos.iteritems(),key=operator.itemgetter(1))
        dictionaryListNeg = sorted(neg.iteritems(),key=operator.itemgetter(1))

        for key, value in dictionaryListInf:
            inft.writerow([str(key),str(value)])
        for key, value in dictionaryListPos:
            post.writerow([str(key),str(value)])
        for key, value in dictionaryListNeg:
            negt.writerow([str(key),str(value)])

    def checkNeutral(self,n):
        tokens  = WhitespaceTokenizer().tokenize(n)
        tagged = nltk.pos_tag(tokens)
        #namedEnt = nltk.ne_chunk(tagged, binary=True)
        print "Neautral Tags:", tagged

    def selectFeatures(self, t,s):
        wholet = ' '.join(t)
        gems = self.checkIfAdjectivev2(wholet)
        return gems

    def getWords(self,tweets):
        all_words = []

        for (words, sentiment) in tweets:

            #print "Entities:", self.checkIfAdjectivev2(wholedata)
            if sentiment == 'positive' or sentiment == 'negative' or sentiment == 'neutral':
                g = self.selectFeatures(words, sentiment)
                #just to build all the words
                if sentiment == "positive":
                    self.poswords.extend(words)
                elif sentiment == "negative":
                    self.negwords.extend(words)
                else:
                    self.infwords.extend(words)


            if not g:
                continue
            else:
                all_words.extend(g)#gets a list of all the words in a tweet

        #print all_words
        return all_words

    def featureExtractor(self,document):
        document_words = set(document)
        features = {}
        for word in self.word_features:
            features['has(%s)' % word] = (word in document_words)

        return features

    def myClassifier(self,tweets,test_tweets):
        r = open('logs.txt','a')

        training_set = nltk.classify.apply_features(self.featureExtractor,tweets)

        algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]

        # Naive Baysian Classifier
        self.classifier = nltk.NaiveBayesClassifier.train(training_set)

        #MAX ENT Classifier
        #classifier = nltk.MaxentClassifier.train(training_set, algorithm,max_iter=3)

        #SVM classifier
        #classifier = nltk.classify.SklearnClassifier(LinearSVC())
        #classifier.train(training_set)

        testset = nltk.classify.apply_features(self.featureExtractor,test_tweets)


        line2 = "Test Tweets:"+str(test_tweets.__len__())+'\n' + "Training Tweets:"+ str(tweets.__len__())+"\n" \
                + "Classifier Accuracy:"+ str(nltk.classify.accuracy(self.classifier,testset))+"\n"
        print line2

        #This calculates the precission, recall and Fmeasure
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        for i, (feats, label) in enumerate (testset):
            refsets[label].add(i)
            observed = self.classifier.classify(feats)
            testsets[observed].add(i)

        print "POS precision:", nltk.metrics.precision(refsets['positive'],testsets['positive'])
        print "POS recall:", nltk.metrics.recall(refsets['positive'],testsets['positive'])
        print "POS F-measure:", nltk.metrics.f_measure(refsets['positive'],testsets['positive'])

        print "NEG precision:", nltk.metrics.precision(refsets['negative'],testsets['negative'])
        print "NEG recall:", nltk.metrics.recall(refsets['negative'],testsets['negative'])
        print "NEG f-measure", nltk.metrics.f_measure(refsets['negative'],testsets['negative'])

        print "INF precision:", nltk.metrics.precision(refsets['neutral'],testsets['neutral'])
        print "INF recall:", nltk.metrics.recall(refsets['neutral'],testsets['neutral'])
        print "INF f-measure:", nltk.metrics.f_measure(refsets['neutral'],testsets['neutral'])
        #r.write(line2)

        test_truth = [s for (t,s) in testset]
        test_predict = [self.classifier.classify(t) for (t,s) in testset]

        print "Confusion Matrix"
        print nltk.ConfusionMatrix(test_truth,test_predict)

        #r.write(str(classifier.show_most_informative_features(10)))

        shuffle(test_tweets)
        #print test_tweets.pop()

        #Runs the demo. (need to uncomment the tweet classifier call.)
        #self.demoClassifier(self.classifier)

        '''
        for n in range (1,30):
            trytweet = test_tweets.pop()
            test = ' '.join(trytweet[0])

            line3 = str(test)+'\t'+"Label:"+str(trytweet[1])+'\t'+ "Predicted Label:"+str(classifier.classify(self.featureExtractor(test.split())))+"\n"
            r.write(line3)

        r.close()
        '''
    def cutList(self,tweet,size):
        final = []
        indicator = 0
        for row in tweet:
            #print row[0] , row[1]
            whole = tuple(row[0]+row[1])
            #whole=(row[0],)
            #whole+=(row[1])
            final.append((row[0],row[1]))
            indicator+=1
            if (indicator == size):
                break
        return final

    #def demoClassifier(self,classifier):


class listener(StreamListener):
        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(212)
        
        def animate(i):
            x_list=[]
            labels_list = ['positive','negative']
            pulldata = open('plots.txt','r').read()
            dataArray = pulldata.split('\n')
            xar = []
            yar = []
            for eachLine in dataArray:
                if len(eachLine)>1:
                    x_list = []
                    x,y = eachLine.split(',')
                    x_list.append(int(x))
                    x_list.append(int(y))

            #global ax1
            ax1.clear()
            #ax1.plot (5,4)
            ax1.pie(x_list,labels=labels_list,autopct='%1.1f%%')

        global fig
        ani = animation.FuncAnimation(fig, animate, interval=1000)
        plt.show()
        '''

        def on_data(self, data):
            try:
                    #print data
                #writeplots = open('plots.txt','w')
                #langauge = data.split(',"lang":"')[1].split('","contributors_enabled')[0]
                #langauge = data.split(',"lang":"')[1].split('","contributors_enabled')[0]
                convtweet = []
                convtweet = json.loads(data)
                if convtweet['lang']:
                    if convtweet['coordinates'] and convtweet['lang'] == "en":
                        textoftweet = convtweet[u'text']
                        tweet = new.tweet_reduction(textoftweet)
                        labelpredicted = sentclassifier.classifier.classify(sentclassifier.featureExtractor(tweet.split()))

                        if labelpredicted == "neutral":
                                labeltowrite = 'neutral'
                        elif labelpredicted == "negative":
                                labeltowrite = 'negative'
                                global globalnegative
                                globalnegative+=1
                        else:
                                labeltowrite = 'positive'
                                global globalpositive
                                globalpositive+=1


                        saveThistofile = open('testhongkong2.csv','a')
                        sthiscsv = csv.writer(saveThistofile,delimiter=',')

                        sthiscsv.writerow([str(convtweet['coordinates']['coordinates'][0]), \
                        str(convtweet['coordinates']['coordinates'][1]), \
                        str(convtweet['created_at']),\
                        textoftweet.encode('utf8'),str(labeltowrite)])
                return True
            except ValueError:
                print "Opps failure to continue running"

        def on_error(self, status):
            print status

########################################################################################################
#main Program
########################################################################################################
ckey = 'YOUR CONSUMER KEY'
csecret = 'YOUR CONSUMER SECRET'
atoken = 'YOUR ACCESS TOKEN'
asecret = 'YOUR ACCESS SECRET'


new = featureReduction() #creates the necessary processing for feature extraction
sentclassifier = nbClassifier() #calls the classifier

####
### This sections pull tweets from Twitter Public API to test classifier.
###

'''
auth = OAuthHandler(ckey, csecret)

auth.set_access_token(atoken, asecret)
l=listener()
twitterStream = Stream(auth, l)
twitterStream.filter(track=["google"])  
'''
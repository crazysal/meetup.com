import string
import nltk 
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin

import time  
import csv
import numpy as np
 
import pickle 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')



# #### Define the Text Computation Pre Processor Class
# ###### The transform method 
# takes a list of documents (given as the variable, X) and returns a new list of tokenized documents, where each document is transformed into list of ordered tokens.
#     
# ###### The tokenize method 
# breaks raw strings into sentences, then breaks those sentences into words and punctuation, 
# and applies a part of speech tag. The token is then normalized: made lower case,
# then stripped of whitespace and other types of punctuation that may be appended. 
# If the token is a stopword or if every character is punctuation, the token is ignored. 
# If it is not ignored, the part of speech is used to lemmatize the token, which is then yielded
# 
#                 
# ###### The Lemmatization method 
# is the process of looking up a single word form from the variety of morphologic affixes that can be applied to indicate tense, plurality, gender, etc. First we need to identify the WordNet tag form based on the Penn Treebank tag, which is returned from NLTK’s standard pos_tag function. 
# We simply look to see if the Penn tag starts with ‘N’, ‘V’, ‘R’, or ‘J’ and can correctly identify if its a noun, verb, adverb, or adjective. We then use the new tag to look up the lemma in the lexicon.


class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = stopwords or set(sw.words('english'))
        self.punct      = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]
    
    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma
                
    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


# #### Build and Evaluate
# ###### The Build Method  
# - takes a classifier class or instance (if given a class, it instantiates the classifier with the defaults) and creates the pipeline with that classifier and fits it. 
# - The function times the build process, evaluates it via the classification report that reports precision, recall, and F1. 
# - Then builds a new model on the complete dataset and writes it out to disk
# 
# *Note that when using the TfidfVectorizer you must make sure that its default preprocessor, normalizer, and tokenizer are all turned off using the identity function and passing None to the other parameters.*


# decorator function to time functions

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time() - ts
#         print('[TimeIt] func: "{}" run in {}s'.format(method.__name__, te - ts))
        return result, te 
    return timed

# identity function as preprocessing of text is being done by NLTKPreprocessor 
def identity(x):
    return x

@timeit
def build_and_evaluate(X, y, classifier=SGDClassifier, outpath=None, verbose=True):

    @timeit
    def build(classifier, X, y=None):
        """
        Inner build function that builds a single model.
        """
        if isinstance(classifier, type):
            classifier = classifier()

        model = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
            ('classifier', classifier),
        ])
        model.fit(X, y)
        return model

    # Label encode the targets
    labels = LabelEncoder()
    y = labels.fit_transform(y)

    # Begin evaluation
    if verbose: 
        print("Building for evaluation")
        
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    model,secs = build(classifier, X_train, y_train)
    if verbose:
        print("Evaluation test model fit in {:0.3f} seconds".format(secs))
        print("Classification Report:\n")
    y_pred = model.predict(X_test)
    print(clsr(y_test, y_pred, target_names=labels.classes_))
    if verbose:
        print("Building complete model and saving ...")
    model, secs = build(classifier, X, y)
    model.labels_ = labels
    if verbose:
        print("Complete model fit in {:0.3f} seconds".format(secs))
        
    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)
        print("Model written out to {}".format(outpath))
    return model

 

# 
# ###### The next stage is to create the pipeline, train a classifier, then to evaluate it
# - The model is split into a training and testing set by shuffling the data
# - The model is trained on the training set, and evaluated on testing.
# - A new model is then fit on all of the data and saved to disk.

# ##### Concatenating textual data from groups and events  
# prepare_data takes in the original csv and divides into train, test, validate sets for further use. 
def prepare_data(csv_data):
	mp = csv_data_df[1:] #1st row was column names
	meetup_df = pd.DataFrame(mp, columns=csv_data_df[0])
	meetup_df['event_X'] = meetup_df['group_name'].map(str) +" " +  meetup_df['category_name'].map(str) +' ' +meetup_df['venue_name'].map(str) +" " + meetup_df['group_description'].map(str) +" " +  meetup_df['event_description'].map(str)+" " +  meetup_df['bio'].map(str)      
	event_X = meetup_df.as_matrix(['event_X']).ravel() 
	event_Y = meetup_df.as_matrix(['event_name']).ravel()
	X_training, X_Validation, Y_training, Y_validation = tts(event_X, event_Y, test_size=0.1)
	return X_training, X_Validation, Y_training, Y_validation

def train_model(x_t, y_t, op):
    model = build_and_evaluate(x_t, y_t, outpath=op)

def validate_model(x_v, y_v, op):
    with open(op, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_Validation)
    print(clsr(Y_validation, y_pred, target_names=labels.classes_))
    

reader = csv.reader(open("../data/meetup_sane.csv"), delimiter=",")
meetup = list(reader) 
x_training, x_validation, y_training, y_validation = prepare_data(meetup)
PATH = "simple_ml_op/model.pickle"

# Run following function to train and save model
train_model(x_training, y_training, PATH)

# Run following function to validate saved model
validate_model(x_validation, y_validation, PATH)



 




  



 

import pandas as pd 
from bs4 import BeautifulSoup
import re
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import word2vec

#nltk.download()
train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter='\t', quoting=3)
unlabeled = pd.read_csv("data/unlabeledTrainData.tsv", header=0, delimiter='\t', quoting=3)
reviews = train['review'].values.tolist()
pattern = r'[^a-zA-z0-9/s]'
stopwords = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def clean_review(review):
	review = BeautifulSoup(review).get_text()
	review = re.sub(pattern, ' ', review)	
	tokenized = word_tokenize(review)
	tokenized = [w.lower() for w in tokenized]
	filtered = [t.strip() for t in tokenized if t not in stopwords]
	return filtered

def make_sentences(review):
	raw_sentences = tokenizer.tokenize(review.strip())
	all_sentences = []

	for sentence in raw_sentences:
		if len(sentence) > 0:
			all_sentences.append(clean_review(sentence))
	return all_sentences

def create_sentence_list():
	sentence_list = []
	i = 0
	for review in train['review']:
		print(i)
		sentence_list += make_sentences(review)
		i += 1
	for review in unlabeled['review']:
		print(i)
		sentence_list += make_sentences(review)
		i += 1
	return sentence_list

def create_model(sentences):
	num_workers = 4
	num_features = 200
	min_words = 30
	num_window = 5

	model = word2vec.Word2Vec(sentences=sentences, size=num_features, workers=num_workers, window=num_window, min_count=min_words, sample=1e-3)

	model.init_sims(replace=True)

	model.wv.save_word2vec_format("model_two.txt", binary = False)

	print(model.most_similar("man"))

if __name__ == '__main__':
	sentences = create_sentence_list()
	create_model(sentences)

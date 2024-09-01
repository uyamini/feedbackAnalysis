import nltk
nltk.data.path.append('/Users/yux9036/nltk_data')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

nltk.download('punkt')

sample_text = "Hello world! This is a test."
tokens = word_tokenize(sample_text)
print(tokens)
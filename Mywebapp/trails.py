import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
print((len(stopwords.words('english'))))
for x in
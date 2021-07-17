import pandas as pd
from bs4 import BeautifulSoup
import re
from sklearn.metrics import r2_score
from catboost import CatBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import classification_report
#from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
#from wordcloud import WordCloud
#from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC,LinearSVC
#from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
#from xgboost import XGBClassifier
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.metrics import accuracy_score
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.experimental import enable_hist_gradient_boosting
#from sklearn.ensemble import HistGradientBoostingClassifier
#from imblearn.over_sampling import SMOTE
#import plotly.express as px
data_set=pd.read_csv("mbti_1.csv")
print(data_set.describe(include="O"))
data_set["posts"][0]


# In[16]:


def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    return text


# In[17]:


count=0
for i in range(8675):
    if data_set["posts"][i].count("|||")!= 49:
        print("this one",i,"====>",data_set["posts"][i].count("|||"))
        count+=1
print("done",count)


# In[18]:


data_set["posts"][774]


# In[19]:


data_set['clean_posts'] = data_set['posts'].apply(cleanText)


# In[20]:


data_set['clean_posts'][0]


# In[21]:


train_data,test_data=train_test_split(data_set,test_size=0.2,random_state=42,stratify=data_set.type)


# In[22]:


train_data


# In[23]:


class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word)>2]


# In[24]:


vectorizer=TfidfVectorizer( max_features=5000,stop_words='english',tokenizer=Lemmatizer())
vectorizer.fit(train_data.posts)


# In[25]:


train_post=vectorizer.transform(train_data.posts).toarray()
test_post=vectorizer.transform(test_data.posts).toarray()


# In[26]:


train_post.shape


# In[27]:


target_encoder=LabelEncoder()
train_target=target_encoder.fit_transform(train_data.type)
test_target=target_encoder.fit_transform(test_data.type)


# In[ ]:


model_cat=CatBoostClassifier(loss_function='MultiClass',eval_metric='MultiClass',task_type='GPU',verbose=False)
model_cat.fit(train_post,train_target)
y=model_cat.predict(test_target)


# In[ ]:





# In[ ]:





# In[ ]:





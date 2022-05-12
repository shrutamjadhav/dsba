DSBDA Experiment 1 
 
 
 
 
1.
 
#Imported required Libraries
import numpy as np
import pandas as pd
#Loading the dataset into pandas dataframe
missing_values = ["Na", "na"]
df=pd.read_csv("StudentsPerformanceTest1.csv", na_values = missing_values)
df.head() 
 
 
 
 
2.
#Checking Null Values
df.isnull().sum() 
 
 
 
 
3.
 
 
#Handling Null Values with mean()
df['math score'] = df['math score'].fillna(df['math score'].mean())
df['reading score'] = df['reading score'].fillna(df['reading score'].mean())
df['writing score'] = df['writing score'].fillna(df['writing score'].mean())
df['Placement Score'] = df['Placement Score'].fillna(df['Placement Score'].mean())
df.dropna(inplace = True) #Clearing all entries with Null values for regions
df.head() 
 
 
 
 
4.
 
 
#Summarising the Datatypes
df.dtypes 
 
 
 
 
5.
 
 
 
 
#Turning Categorical Variables into Numerical Values
 
df['gender'].replace(to_replace='female', value=1, inplace = True)
df['gender'].replace(to_replace='male', value=2, inplace = True)
df.head()
 
 
 
 
DSBDA Experiment 2
 
 
 
 
1.
 
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
missing_values = ["Na", "na"]
df=pd.read_csv("StudentsPerformanceTest1.csv", na_values = missing_values)
df.head()
 
 
 
 
2.
 
df.describe() 
 
 
 
 
3.
 
#Check for null values
df.isnull().sum() 
 
 
 
 
4.
 
 
#Handling Null Values with Mean
df['math score'] = df['math score'].fillna(df['math score'].mean())
df['reading score'] = df['reading score'].fillna(df['reading score'].mean())
df['writing score'] = df['writing score'].fillna(df['writing score'].mean())
df['Placement Score'] = df['Placement Score'].fillna(df['Placement Score'].mean())
df.dropna(axis = 1,inplace = True) #Dropping Region Coloumn 
df.head() 
 
 
 
 
5.
 
 
#Showing Outliers using Boxplot
col = ['math score', 'reading score' , 'writing score','Placement Score']
df.boxplot(col) 
 
 
 
 
6.
 
 
#Label Encoding for categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df.head() 
 
 
 
 
DSBDA Experiment 3
 
 
 
 
1.
 
df.mean()
 
 
 
 
2.
 
 
df.mean() 
 
 
 
 
3.
 
 
df.loc[:,'Age'].mean() 
 
 
 
 
Mean
 
4.
 
 
df.mean(axis=1)[0:4] 
 
 
 
 
5.
 
df.median()
 
 
 
 
6.
 
df.loc[:,'Age'].median() 
 
 
 
 
Median
 
7.
 
 
df.median(axis=1)[0:4] 
 
 
 
 
Mode
 
 
 
 
8.
 
 
df.mode()
 
 
 
 
9.
 
 
 
 
 
df.loc[:,'Age'].mode() 
 
 
 
 
10.
 
 
 
 
 
df.min()
 
 
 
 
11.
 
 
 
 
df.loc[:,'Age'].min(skipna = False) 
 
 
 
 
12.
 
 
df.max() 
 
 
 
 
13.
 
 
 
 
 
df.loc[:,'Age'].max(skipna = False) 
 
 
 
 
14.
 
 
 
 
 
df.std()
 
 
 
 
15.
 
 
 
 
 
df.loc[:,'Age'].std() 
 
 
 
 
16.
 
 
 
 
 
df.std(axis=1)[0:4] 
 
 
 
 
Standerd Deviation
 
17.
 
df.groupby(['Genre'])['Age'].mean() 
 
 
 
 
 
 
 
18.
 
df_u = df.rename(columns = {'Annual Income (k$)':'Income'}, inplace=False)
(df_u.groupby(['Genre']).Income.mean())
 
 
 
 
 
 
 
19.
 
 
df_u 
 
 
 
 
GroupBy
 
20.
 
df_u.groupby(['Genre']).Income.mean() 
 
 
 
 
Sklearn
 
21.
 
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(df[['Genre']]).toarray())
enc_df 
 
 
 
 
22.
 
 
df_encode = df_u.join(enc_df)
df_encode 
 
 
 
 
DSBDA Experiment 4
 
 
 
 
1.
 
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
 
 
 
 
2.
 
 
x = np.array([95,85,80,70,60])
y = np.array([85,95,70,65,70])
 
 
 
 
3.
 
model = np.polyfit(x,y,1)
 
 
 
 
4.
 
 
model 
 
 
 
 
5.
predict = np.poly1d(model)
predict(65) 
 
 
 
 
6.
y_pred = predict(x)
y_pred 
 
 
 
 
using sklearn
 
 
 
 
7.
 
 
from sklearn.metrics import r2_score
r2_score(y, y_pred)
 
 
 
 
n [8]:
y_line = model[1] + model[0]* x
plt.plot(x, y_line, c = 'r')
plt.scatter(x, y_pred)
 
 
 
 
In [9]:
pip install sklearn
 
 
 
 
Boston Dataset 
 
In [10]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
 
 
 
In [11]:
pip install sklearn
 
 
 
 
In [12]:
df = pd.read_csv("housing.csv")
df 
 
 
 
 
In [13]:
from sklearn.datasets import load_boston
boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data.head()
 
 
 
 
In [14]:
data['PRICE'] = boston.target
data.isnull().sum()
x = data.drop(['PRICE'], axis = 1)
y = data['PRICE'] 
 
 
 
 
In [16]:
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2,random_state = 0)
 
 
 
 
In [18]:
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(xtrain, ytrain)
 
 
 
 
In [19]:
ytrain_pred = lm.predict(xtrain)
ytest_pred = lm.predict(xtest)
 
 
 
 
In [20]:
df=pd.DataFrame(ytrain_pred,ytrain)
df=pd.DataFrame(ytest_pred,ytest)
 
 
 
 
In [21]:
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(ytest, ytest_pred)
print(mse)
mse = mean_squared_error(ytrain_pred,ytrain)
print(mse)
 
 
 
 
In [25]:
plt.scatter(ytrain ,ytrain_pred,c='blue',marker='o',label='Training data')
plt.scatter(ytest,ytest_pred ,c='lightgreen',marker='s',label='Test data')
plt.xlabel('True values')
plt.ylabel('Predicted')
plt.title("True value vs Predicted value")
plt.legend(loc= 'upper left') #plt.hlines(y=0,xmin=0,xmax=50)
plt.plot()
plt.show()
 
 
 
 
 
Conclusion: 
 
In [27]:
# In this way we have done data analysis using linear regression
# for Boston Dataset and predict the price of houses using the
# features of the Boston Dataset.
 
 
 
 
DSBDA Experiment 5
 
In [1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
 
 
 
 
In [2]:
df = pd.read_csv("Social_Network_Ads.csv") 
 
 
 
 
In [3]:
df.head() 
 
 
 
 
In [4]:
df.shape
 
 
 
 
Visualisation
 
In [6]:
sns.displot(df['Age'])
 
 
 
 
In [8]:
sns.displot(df['EstimatedSalary'])
 
 
 
 
split data into independent and dependent value
 
In [10]:
X = np.asarray(df[['Age', 'EstimatedSalary']])
Y = np.asarray(df['Purchased'])
 
 
 
 
By Ma'am Method
 
In [12]:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset.head() 
 
 
 
 
In [13]: X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
 
print(X[:3, :])
print('-'*15)
print(y[:3]) 
 
 
 
 
 
In [14]:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train[:3])
print('-'*15)
print(y_train[:3])
print('-'*15)
print(X_test[:3])
print('-'*15)
print(y_test[:3])
 
 
 
 
In [15]:
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 
 
 
 
 
In [16]:
print(X_train[:3])
print('-'*15)
print(X_test[:3])
 
 
 
 
In [17]:
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs' )
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(X_test[:10])
print('-'*15)
print(y_pred[:10])
 
 
 
 
 
In [18]:
print(y_pred[:20])
print(y_test[:20]) 
 
 
 
 
 
 
 
In [19]:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
 
 
 
 
In [20]:
# Visualizing the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, ste
 np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, ste
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
 alpha = 0.6, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
 plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
 c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() 
 
 
 
 
DSBA EXP 6:
 
 
 
 
1.import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('Iris.csv')
print(df.shape)
df 
 
 
 
 
2.
 
print('The DataFrame Contains %d Rows and %d Columns '%(df.shape))
print(df.dtypes)
print(df.info())
print(df.head())
 
 
 
 
3.
 
features = df.iloc[:,0:4]
print(features.head())
 
 
 
 
 
4.
 
target=df.iloc[:,5]
print(target) 
 
 
 
 
5.
 
print('The Initial DataFrame Contained %d Rows and %d Columns'%(df.shape))
print('The Features Matrix Contains %d Rows And %d Columns'%(features.shape))
print('The Target Vector Contains %d Rows And %d Columns'%(np.array(target).reshape(-1,1).shape))
 
 
 
 
6.
 
from sklearn.naive_bayes import GaussianNB
algorithm=GaussianNB(priors=None, var_smoothing=1e-9)
algorithm.fit(features,target) 
 
 
 
 
7.
 
print(algorithm.classes_)
 
 
 
 
8.
 
 
 
 
print('The Gaussian Model Has Achieved %.2f Percent Accuracy'%(algorithm.score(features,target)))
 
 
 
 
9.
 
 
 
 
observation=[[5.0,3.7,1.6,0.1]]
predictions=algorithm.predict(observation)
print(predictions)
 
 
 
 
DSBA EXP 7A
 
 
 
 
In [1]:
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
 
 
 
 
In [2]:
text= "Tokenization is the first step in text analytics. The process of breaking down a text pa 
 
 
 
 
In [3]:
#Sentence Tokenization
tokenized_text= sent_tokenize(text)
print(tokenized_text)
#Word Tokenization
tokenized_word=word_tokenize(text)
 
 
 
 
In [4]:
#Print stop words of English
stop_words=set(stopwords.words("english"))
print(stop_words)
text= "How to remove stop words with NLTK library in Python?"
text= re.sub('[^a-zA-Z]', ' ',text)
tokens = word_tokenize(text.lower())
filtered_text=[]
for w in tokens:
 if w not in stop_words:
 filtered_text.append(w)
print("Tokenized Sentence:",tokens)
print("Filterd Sentence:",filtered_text) 
print(tokenized_word) 
 
 
 
 
In [5]:
#Stemming
e_words= ["wait", "waiting", "waited", "waits"]
ps =PorterStemmer()
for w in e_words:
 
 
rootWord=ps.stem(w)
print(rootWord) 
 
 
 
 
In [6]:
#Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
 print("Lemma for {} is {}".format(w,wordnet_lemmatizer.lemmatize(w)))
 
 
 
 
In [7]:
#Pos Tagging
data="The pink sweater fit her perfectly"
words=word_tokenize(data)
for word in words:
 print(nltk.pos_tag([word])) 
 
 
 
 
Assignment 7B 
 
 
 
 
In [8]:
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math
 
In [9]:
documentA = 'Jupiter is the largest Planet'
documentB = 'Mars is the fourth planet from the Sun'
bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')
uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))
numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
 numOfWordsA[word] += 1
 numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
 numOfWordsB[word] += 1
 
 
In [10]:
def computeTF(wordDict, bagOfWords):
 tfDict = {}
 bagOfWordsCount = len(bagOfWords)
 for word, count in wordDict.items():
 tfDict[word] = count / float(bagOfWordsCount)
 return tfDict
tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)
 
In [11]:
def computeIDF(documents):
 
 N = len(documents)
 idfDict = dict.fromkeys(documents[0].keys(), 0)
 for document in documents:
 for word, val in document.items():
 if val > 0:
 
In [11]:
def computeIDF(documents):
 
 N = len(documents)
 idfDict = dict.fromkeys(documents[0].keys(), 0)
 for document in documents:
 for word, val in document.items():
 if val > 0:
 
In [12]:
def computeTFIDF(tfBagOfWords, idfs):
 tfidf = {}
 for word, val in tfBagOfWords.items():
 tfidf[word] = val * idfs[word]
 return tfidf
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
df = pd.DataFrame([tfidfA, tfidfB])
df
 
DSBA EXP 8
 
Importing Necessary Libraries
 
In [1]:
import numpy as np
port pandas as pa
import matplotlib. pyplot as plt
(matplotlib inline
import warntones
warnings. filterwarnings( 'ignore' )
 
Loading the dataset
 
In [2]:
titanic = sns. load_dataset( 'titanic' )
 
In [3]:
titanic. info( )
 
 
n [4]:
titanic. describe()
 
 
In [5]:
 
titanic . shape
out [5] :
(891, 15)
 
Data Visualization
In [6]:
sns . histplot(x=' fare' , data=titanic)
sns. set (rc={' figure. figsize' :(5,5)})
 
In [7]:
sns . displot(x='age' , data=titanic, bins=70)
sns. set (rc={ ' figure. figsize' : (5,5)})
 
In [8]:
sns. factorplot( 'survived' , data=titanic, kind=' count' , hue='pclass' )
sns. set(rc={'figure. figsize' : (5,5)})
 
In [9]:
sns. factorplot( ' survived' , data=titanic, kind=' count' , hue='sex' )
sns . set (rc={ 'figure. figsize' : (5,5)})
 
DSBDA Exp 9
 
1.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset = sns.load_dataset('titanic')
dataset
 
2.
sns.boxplot(x='sex',y='age',data=dataset)
 
3.
#Boxplot
sns.boxplot(x=dataset['sex'],y=dataset['age'],hue=dataset['survived'] )
 
 
DSBDA EXP 10
 
1.
Importing the required libraries:
In [1]:
import numpy as up
import pandas as pd
import matplotlib. pyplot as pit
import seaborn as sns
 
Here we will load the Iris dataset and store it in a dataframe using pandas.
in [2]:
dataset = sns. load_dataset('iris')
 
dataset = sns. load_dataset('iris')
 
In [3]:
dataset . head( )
 
In [4]:
dataset. info( )
 
In [5]:
fig, axes = plt. subplots(2, 2, figsize=(16, 8))
 
axes [0,0]. set_title("Distribution of Sepal Length")
axes[0,0]. hist(dataset ["sepal_length"]);
axes[0, 1]. set_title("Distribution of Sepal Width")
axes [0, 1]. hist (dataset ["sepal_width"]);
 
axes [1,0]. set_title("Distribution of Petal Length")
axes [1, 0]. hist(dataset ["petal_length"]);
axes [1, 1]. set_title("Distribution of Petal Width")
axes [1, 1]. hist(dataset ["petal_width"]);
 
We plot the Boxplot for each feature in the dataset.
 
In [7]:
fig, axes = plt. subplots(2, 2, figsize=(16,9) )
axes[0,0] . set_title("Distribution of Sepal Length")
sns . boxplot(y="sepal_length", x= "species", data=dataset, orient='v' , ax=axes[0, 0])
axes[0, 1]. set_title("Distribution of Sepal Length")
sns. boxplot( y="sepal_width", x= "species", data=dataset, orient='v' , ax=axes[0, 1])
axes[1,0]. set_title("Distribution of Sepal Length")
sns. boxplot( y="petal_length", x= "species", data=dataset,orient='v' , ax=axes[1, 0])
axes [1, 1]. set_title("Distribution of Sepal Length")
sns . boxplot( y="petal_width", x= "species", data=dataset, orient='v' , ax=axes[1, 1])
pit. show( )


JustPaste.it
User avatar
@anonymous · 6m · edited: -1s
Hidden 
 
Group B assign
 
1-->
 
MapReduce Word Count Example
In MapReduce word count example, we find out the frequency of each word. Here, the role of Mapper is to map the keys to the existing values and the role of Reducer is to aggregate the keys of common values. So, everything is represented in the form of Key-value pair.
 
Pre-requisite
Java Installation - Check whether the Java is installed or not using the following command.
java -version
Hadoop Installation - Check whether the Hadoop is installed or not using the following command.
hadoop version
If any of them is not installed in your system, follow the below link to install it.
 
 
 
Steps to execute MapReduce word count example
Create a text file in your local machine and write some text into it.
$ nano data.txt
MapReduce Word Count Example
Check the text written in the data.txt file.
$ cat data.txt
MapReduce Word Count Example
In this example, we find out the frequency of each word exists in this text file.
 
Create a directory in HDFS, where to kept text file.
$ hdfs dfs -mkdir /test
Upload the data.txt file on HDFS in the specific directory.
$ hdfs dfs -put /home/codegyani/data.txt /test
MapReduce Word Count Example
Write the MapReduce program using eclipse.
File: WC_Mapper.java
package com.javatpoint;  
  
import java.io.IOException;    
import java.util.StringTokenizer;    
import org.apache.hadoop.io.IntWritable;    
import org.apache.hadoop.io.LongWritable;    
import org.apache.hadoop.io.Text;    
import org.apache.hadoop.mapred.MapReduceBase;    
import org.apache.hadoop.mapred.Mapper;    
import org.apache.hadoop.mapred.OutputCollector;    
import org.apache.hadoop.mapred.Reporter;    
public class WC_Mapper extends MapReduceBase implements Mapper<LongWritable,Text,Text,IntWritable>{    
    private final static IntWritable one = new IntWritable(1);    
    private Text word = new Text();    
    public void map(LongWritable key, Text value,OutputCollector<Text,IntWritable> output,     
           Reporter reporter) throws IOException{    
        String line = value.toString();    
        StringTokenizer  tokenizer = new StringTokenizer(line);    
        while (tokenizer.hasMoreTokens()){    
            word.set(tokenizer.nextToken());    
            output.collect(word, one);    
        }    
    }    
    
}  
File: WC_Reducer.java
package com.javatpoint;  
    import java.io.IOException;    
    import java.util.Iterator;    
    import org.apache.hadoop.io.IntWritable;    
    import org.apache.hadoop.io.Text;    
    import org.apache.hadoop.mapred.MapReduceBase;    
    import org.apache.hadoop.mapred.OutputCollector;    
    import org.apache.hadoop.mapred.Reducer;    
    import org.apache.hadoop.mapred.Reporter;    
        
    public class WC_Reducer  extends MapReduceBase implements Reducer<Text,IntWritable,Text,IntWritable> {    
    public void reduce(Text key, Iterator<IntWritable> values,OutputCollector<Text,IntWritable> output,    
     Reporter reporter) throws IOException {    
    int sum=0;    
    while (values.hasNext()) {    
    sum+=values.next().get();    
    }    
    output.collect(key,new IntWritable(sum));    
    }    
    }  
File: WC_Runner.java
package com.javatpoint;  
  
    import java.io.IOException;    
    import org.apache.hadoop.fs.Path;    
    import org.apache.hadoop.io.IntWritable;    
    import org.apache.hadoop.io.Text;    
    import org.apache.hadoop.mapred.FileInputFormat;    
    import org.apache.hadoop.mapred.FileOutputFormat;    
    import org.apache.hadoop.mapred.JobClient;    
    import org.apache.hadoop.mapred.JobConf;    
    import org.apache.hadoop.mapred.TextInputFormat;    
    import org.apache.hadoop.mapred.TextOutputFormat;    
    public class WC_Runner {    
        public static void main(String[] args) throws IOException{    
            JobConf conf = new JobConf(WC_Runner.class);    
            conf.setJobName("WordCount");    
            conf.setOutputKeyClass(Text.class);    
            conf.setOutputValueClass(IntWritable.class);            
            conf.setMapperClass(WC_Mapper.class);    
            conf.setCombinerClass(WC_Reducer.class);    
            conf.setReducerClass(WC_Reducer.class);         
            conf.setInputFormat(TextInputFormat.class);    
            conf.setOutputFormat(TextOutputFormat.class);           
            FileInputFormat.setInputPaths(conf,new Path(args[0]));    
            FileOutputFormat.setOutputPath(conf,new Path(args[1]));     
            JobClient.runJob(conf);    
        }    
    }    
Download the source code.
Create the jar file of this program and name it countworddemo.jar.
Run the jar file
hadoop jar /home/codegyani/wordcountdemo.jar com.javatpoint.WC_Runner /test/data.txt /r_output
The output is stored in /r_output/part-00000
MapReduce Word Count Example
Now execute the command to see the output.
hdfs dfs -cat /r_output/part-00000
 
 
2-->
 
package SalesCountry;
 
import java.io.IOException;
import java.util.*;
 
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
 
public class SalesCountryReducer extends MapReduceBase implements Reducer<Text, IntWritable, Text, IntWritable> {
 
public void reduce(Text t_key, Iterator<IntWritable> values, OutputCollector<Text,IntWritable> output, Reporter reporter) throws IOException {
Text key = t_key;
int frequencyForCountry = 0;
while (values.hasNext()) {
// replace type of value with the actual type of our value
IntWritable value = (IntWritable) values.next();
frequencyForCountry += value.get();
 

}
output.collect(key, new IntWritable(frequencyForCountry));
}
}
 
3-->
 
package com.org.vasanth.weather;
 
 
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;
 
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.KeyValueTextInputFormat;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
 
/**
 * @author Vasanth Mahendran
 * 
 * This is an Hadoop Map/Reduce application for Working on weather data It reads
 * the text input files, breaks each line into stations weather data and finds
 * average for temperature , dew point , wind speed. The output is a locally
 * sorted list of stations and its 12 attribute vector of average temp , dew ,
 * wind speed of 4 sections for each month.
 *
 * To run: bin/hadoop jar target/weather-1.0.jar [-m <i>maps</i>] [-r
 * <i>reduces</i>] <i>in-dir for job 1</i> <i>out-dir for job 1</i> <i>out-dir
 * for job 2</i>
 */
public class Weather extends Configured implements Tool {
final long DEFAULT_SPLIT_SIZE = 128 * 1024 * 1024;
 
/**
* Map Class for Job 1
* 
* For each line of input, emits key value pair with
* station_yearmonth_sectionno as key and 3 attribute vector with
* temperature , dew point , wind speed as value.Map method will strip the
* day and hour from field and replace it with section no (
* <b>station_yearmonth_sectionno</b>, <b><temperature,dew point , wind
* speed></b>).
*/
public static class MapClass extends MapReduceBase
implements Mapper<LongWritable, Text, Text, Text> {
 
private Text word = new Text();
private Text values = new Text();
 
public void map(LongWritable key, Text value,
OutputCollector<Text, Text> output,
Reporter reporter) throws IOException {
String line = value.toString();
StringTokenizer itr = new StringTokenizer(line);
int counter = 0;
String key_out = null;
String value_str = null;
boolean skip = false;
loop:while (itr.hasMoreTokens() && counter<13) {
String str = itr.nextToken();
switch (counter) {
case 0:
key_out = str;
if(str.contains("STN")){//Ignoring rows where station id is all 9
skip = true;
break loop;
}else{
break;
}
case 2:
int hour = Integer.valueOf(str.substring(str.lastIndexOf("_")+1, str.length()));
str = str.substring(4,str.lastIndexOf("_")-2);
/*if(hour<=5){
str = str.concat("_section4");
}else if(hour>5 && hour<=11){
str = str.concat("_section1");
}else if(hour>11 && hour<=17){
str = str.concat("_section2");
}else if(hour>17 && hour<=23){
str = str.concat("_section3");
}*/
 

if(hour>4 && hour<=10){
str = str.concat("_section1");
}else if(hour>10 && hour<=16){
str = str.concat("_section2");
}else if(hour>16 && hour<=22){
str = str.concat("_section3");
}else{
str = str.concat("_section4");
}
 

 

key_out = key_out.concat("_").concat(str);
break;
case 3://Temperature
if(str.equals("9999.9")){//Ignoring rows where temperature is all 9
skip = true;
break loop;
}else{
value_str = str.concat(" ");
break;
}
case 4://Dew point
if(str.equals("9999.9")){//Ignoring rows where dew point is all 9
skip = true;
break loop;
}else{
value_str = value_str.concat(str).concat(" ");
break;
}
case 12://Wind speed
if(str.equals("999.9")){//Ignoring rows where wind speed is all 9
skip = true;
break loop;
}else{
value_str = value_str.concat(str).concat(" ");
break;
}
default:
break;
}
counter++;
}
if(!skip){
word.set(key_out);
values.set(value_str);
output.collect(word, values);
}
}
}
 

/**
* Map Class for Job 2
* 
* For each input, emits key value pair with station_yearmonth as key and 3
* attribute vector with temperature , dew point , wind speed as value by
* stripping the section no from key and adding section no into vector value
* ( <b>station_yearmonth</b>, <b><temperature,dew point , wind speed></b>).
*/
public static class MapClassForJob2 extends MapReduceBase
implements Mapper<Text, Text, Text, Text> {
private Text key_text = new Text();
private Text value_text = new Text();
@Override
public void map(Text key, Text value,
OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
 

String str = key.toString();
String station = str.substring(str.lastIndexOf("_")+1, str.length());
str = str.substring(0,str.lastIndexOf("_"));
key_text.set(str);
 

StringTokenizer itr = new StringTokenizer(value.toString());
String str_out = station.concat("<");
while (itr.hasMoreTokens()) {
String nextToken = itr.nextToken(" ");
str_out = str_out.concat(nextToken);
str_out = ((itr.hasMoreTokens()) ? str_out.concat(",") : str_out.concat(">"));
}
value_text.set(str_out);
output.collect(key_text,value_text);
}
}
 

/**
* Reducer Class for Job 1
* 
* A reducer class that just emits 3 attribute vector with average
* temperature , dew point , wind speed for each of the section of the month
* for each input
*/
public static class Reduce extends MapReduceBase
implements Reducer<Text, Text, Text, Text> {
private Text value_out_text = new Text();
public void reduce(Text key, Iterator<Text> values,
OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
double sum_temp = 0;
double sum_dew = 0;
double sum_wind = 0;
int count = 0;
 

while (values.hasNext()) {
 

String str = values.next().toString();
 

StringTokenizer itr = new StringTokenizer(str);
int count_vector = 0;
while (itr.hasMoreTokens()) {
String nextToken = itr.nextToken(" ");
if(count_vector==0){
sum_temp += Double.valueOf(nextToken);
}
if(count_vector==1){
sum_dew += Double.valueOf(nextToken);
}
if(count_vector==2){
sum_wind += Double.valueOf(nextToken);
}
count_vector++;
}
count++;
}
double avg_tmp = sum_temp / count;
double avg_dew = sum_dew / count;
double avg_wind = sum_wind / count;
 

System.out.println(key.toString()+" count is "+count+" sum of temp is "+sum_temp+" sum of dew is "+sum_dew+" sum of wind is "+sum_wind+"\n");
 

String value_out = String.valueOf(avg_tmp).concat(" ").concat(String.valueOf(avg_dew)).concat(" ").concat(String.valueOf(avg_wind));
value_out_text.set(value_out);
output.collect(key, value_out_text);
}
}
 

/**
* Reducer Class for Job 2
* 
* A reducer class that just emits 12 attribute vector with average
* temperature , dew point , wind speed for all section of the month
* for each input
*/
public static class ReduceForJob2 extends MapReduceBase
implements Reducer<Text, Text, Text, Text> {
private Text value_out_text = new Text();
public void reduce(Text key, Iterator<Text> values,
OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
String value_out = "";
while (values.hasNext()) {
value_out = value_out.concat(values.next().toString()).concat(" ");
}
value_out_text.set(value_out);
output.collect(key, value_out_text);
}
}
 

static int printUsage() {
System.out.println("weather [-m <maps>] [-r <reduces>] <job_1 input> <job_1 output> <job_2 output>");
ToolRunner.printGenericCommandUsage(System.out);
return -1;
}
 
/**
* The main driver for weather map/reduce program.
* Invoke this method to submit the map/reduce job.
* @throws IOException When there is communication problems with the
*                     job tracker.
*/
public int run(String[] args) throws Exception {
Configuration config = getConf();
 

 
// We need to lower input block size by factor of two.
/*config.setLong(
    FileInputFormat.SPLIT_MAXSIZE,
    config.getLong(
        FileInputFormat.SPLIT_MAXSIZE, DEFAULT_SPLIT_SIZE) / 2);*/
 

JobConf conf = new JobConf(config, Weather.class);
conf.setJobName("Weather Job1");
 
// the keys are words (strings)
conf.setOutputKeyClass(Text.class);
// the values are counts (ints)
conf.setOutputValueClass(Text.class);
 

conf.setMapOutputKeyClass(Text.class);
conf.setMapOutputValueClass(Text.class);
 
conf.setMapperClass(MapClass.class);
//conf.setCombinerClass(Combiner.class);
conf.setReducerClass(Reduce.class);
List<String> other_args = new ArrayList<String>();
for(int i=0; i < args.length; ++i) {
try {
if ("-m".equals(args[i])) {
conf.setNumMapTasks(Integer.parseInt(args[++i]));
} else if ("-r".equals(args[i])) {
conf.setNumReduceTasks(Integer.parseInt(args[++i]));
} else {
other_args.add(args[i]);
}
} catch (NumberFormatException except) {
System.out.println("ERROR: Integer expected instead of " + args[i]);
return printUsage();
} catch (ArrayIndexOutOfBoundsException except) {
System.out.println("ERROR: Required parameter missing from " +
args[i-1]);
return printUsage();
}
}
// Make sure there are exactly 2 parameters left.
/*if (other_args.size() != 3) {
System.out.println("ERROR: Wrong number of parameters: " +
other_args.size() + " instead of 3.");
return printUsage();
}*/
FileInputFormat.setInputPaths(conf, other_args.get(0));
FileOutputFormat.setOutputPath(conf, new Path(other_args.get(1)));
 
JobClient.runJob(conf);
 

JobConf conf2 = new JobConf(config, Weather.class);
conf2.setJobName("Weather Job 2");
 
// the keys are words (strings)
conf2.setOutputKeyClass(Text.class);
// the values are counts (ints)
conf2.setOutputValueClass(Text.class);
 

conf2.setInputFormat(KeyValueTextInputFormat.class);
 

conf2.setMapOutputKeyClass(Text.class);
conf2.setMapOutputValueClass(Text.class);
 
conf2.setMapperClass(MapClassForJob2.class);
//conf.setCombinerClass(Combiner.class);
conf2.setReducerClass(ReduceForJob2.class);
 

 
FileInputFormat.setInputPaths(conf2, new Path(other_args.get(1)));
FileOutputFormat.setOutputPath(conf2, new Path(other_args.get(2)));
 
JobClient.runJob(conf2);
return 0;
}
 
 
public static void main(String[] args) throws Exception {
int res = ToolRunner.run(new Configuration(), new Weather(), args);
System.exit(res);
}
 
}
 
4-->
 
What is Apache Spark?
Apache Spark is an Open source analytical processing engine for large scale powerful distributed data processing and machine learning applications. Spark is Originally developed at the University of California, Berkeley’s, and later donated to Apache Software Foundation. In February 2014, Spark became a Top-Level Apache Project and has been contributed by thousands of engineers and made Spark as one of the most active open-source projects in Apache.
 
Apache Spark Features
In-memory computation
Distributed processing using parallelize
Can be used with many cluster managers (Spark, Yarn, Mesos e.t.c)
Fault-tolerant
Immutable
Lazy evaluation
Cache & persistence
Inbuild-optimization when using DataFrames
Supports ANSI SQL
Apache Spark Advantages
Spark is a general-purpose, in-memory, fault-tolerant, distributed processing engine that allows you to process data efficiently in a distributed fashion.
Applications running on Spark are 100x faster than traditional systems.
You will get great benefits using Spark for data ingestion pipelines.
Using Spark we can process data from Hadoop HDFS, AWS S3, Databricks DBFS, Azure Blob Storage, and many file systems.
Spark also is used to process real-time data using Streaming and Kafka.
Using Spark Streaming you can also stream files from the file system and also stream from the socket.
Spark natively has machine learning and graph libraries.
Apache Spark Architecture
Apache Spark works in a master-slave architecture where the master is called “Driver” and slaves are called “Workers”. When you run a Spark application, Spark Driver creates a context that is an entry point to your application, and all operations (transformations and actions) are executed on worker nodes, and the resources are managed by Cluster Manager.
 
apache spark architecture
Source: https://spark.apache.org/
Cluster Manager Types
As of writing this Apache Spark Tutorial, Spark supports below cluster managers:
 
Standalone – a simple cluster manager included with Spark that makes it easy to set up a cluster.
Apache Mesos – Mesons is a Cluster manager that can also run Hadoop MapReduce and Spark applications.
Hadoop YARN – the resource manager in Hadoop 2. This is mostly used, cluster manager.
Kubernetes – an open-source system for automating deployment, scaling, and management of containerized applications.
local – which is not really a cluster manager but still I wanted to mention as we use “local” for master() in order to run Spark on your laptop/computer.
 
Spark Installation
In order to run Apache Spark examples mentioned in this tutorial, you need to have Spark and it’s needed tools to be installed on your computer. Since most developers use Windows for development, I will explain how to install Spark on windows in this tutorial. you can also Install Spark on Linux server if needed.
 
Download Apache Spark by accessing Spark Download page and select the link from “Download Spark (point 3)”. If you wanted to use a different version of Spark & Hadoop, select the one you wanted from drop downs and the link on point 3 changes to the selected version and provides you with an updated link to download.
 
Apache Spark Installation
After download, untar the binary using 7zip and copy the underlying folder spark-3.0.0-bin-hadoop2.7 to c:\apps
 
Now set the following environment variables.
 
!!
SPARK_HOME  = C:\apps\spark-3.0.0-bin-hadoop2.7
HADOOP_HOME = C:\apps\spark-3.0.0-bin-hadoop2.7
PATH=%PATH%;C:\apps\spark-3.0.0-bin-hadoop2.7\bin
!!
 
Python
Setup winutils.exe
Download wunutils.exe file from winutils, and copy it to %SPARK_HOME%\bin folder. Winutils are different for each Hadoop version hence download the right version from https://github.com/steveloughran/winutils
 
spark-shell
Spark binary comes with interactive spark-shell. In order to start a shell, go to your SPARK_HOME/bin directory and type “spark-shell2“. This command loads the Spark and displays what version of Spark you are using.
 
Apache spark shell
spark-shell
By default, spark-shell provides with spark (SparkSession) and sc (SparkContext) object’s to use. Let’s see some examples.
 
spark shell
spark-shell create RDD
Spark-shell also creates a Spark context web UI and by default, it can access from http://localhost:4041.
 
Spark-submit
The spark-submit command is a utility to run or submit a Spark or PySpark application program (or job) to the cluster by specifying options and configurations, the application you are submitting can be written in Scala, Java, or Python (PySpark) code. You can use this utility in order to do the following.
 
Submitting Spark application on different cluster managers like Yarn, Kubernetes, Mesos, and Stand-alone.
Submitting Spark application on client or cluster deployment modes
 
!!
./bin/spark-submit \
  --master <master-url> \
  --deploy-mode <deploy-mode> \
  --conf <key<=<value> \
  --driver-memory <value>g \
  --executor-memory <value>g \
  --executor-cores <number of cores>  \
  --jars  <comma separated dependencies>
  --class <main-class> \
  <application-jar> \
  [application-arguments]
!!
 
Scala
Spark Web UI
Apache Spark provides a suite of Web UIs (Jobs, Stages, Tasks, Storage, Environment, Executors, and SQL) to monitor the status of your Spark application, resource consumption of Spark cluster, and Spark configurations. On Spark Web UI, you can see how the operations are executed.
 
Spark Web UI tutorial
Spark Web UI
Spark History Server
Spark History server, keep a log of all completed Spark application you submit by spark-submit, spark-shell. before you start, first you need to set the below config on spark-defaults.conf
 
!!
spark.eventLog.enabled true
spark.history.fs.logDirectory file:///c:/logs/path
!!
 
Scala
Now, start spark history server on Linux or mac by running.
 
!!
$SPARK_HOME/sbin/start-history-server.sh
!!
 
Scala
If you are running Spark on windows, you can start the history server by starting the below command.
 
!!
$SPARK_HOME/bin/spark-class.cmd org.apache.spark.deploy.history.HistoryServer
!!
Scala
By default History server listens at 18080 port and you can access it from browser using http://localhost:18080/
 
history server
Spark History Server
By clicking on each App ID, you will get the details of the application in Spark web UI.
 
The history server is very helpful when you are doing Spark performance tuning to improve spark jobs where you can cross-check the previous application run with the current run.
 
Spark Modules
Spark Core
Spark SQL
Spark Streaming
Spark MLlib
Spark GraphX
Modules and components
 
 
Spark Core
In this section of the Apache Spark Tutorial, you will learn different concepts of the Spark Core library with examples in Scala code. Spark Core is the main base library of the Spark which provides the abstraction of how distributed task dispatching, scheduling, basic I/O functionalities and etc.
 
Before getting your hands dirty on Spark programming, have your Development Environment Setup to run Spark Examples using IntelliJ IDEA
 
SparkSession
SparkSession introduced in version 2.0, It is an entry point to underlying Spark functionality in order to programmatically use Spark RDD, DataFrame and Dataset. It’s object spark is default available in spark-shell.
 
Creating a SparkSession instance would be the first statement you would write to program with RDD, DataFrame and Dataset. SparkSession will be created using SparkSession.builder() builder pattern.
 
!!
import org.apache.spark.sql.SparkSession
val spark:SparkSession = SparkSession.builder()
      .master("local[1]")
      .appName("SparkByExamples.com")
      .getOrCreate()   
!!
Scala
Spark Context
SparkContext is available since Spark 1.x (JavaSparkContext for Java) and is used to be an entry point to Spark and PySpark before introducing SparkSession in 2.0. Creating SparkContext was the first step to the program with RDD and to connect to Spark Cluster. It’s object sc by default available in spark-shell.
 
Since Spark 2.x version, When you create SparkSession, SparkContext object is by default create and it can be accessed using spark.sparkContext
 
Note that you can create just one SparkContext per JVM but can create many SparkSession objects.
 
Apache Spark RDD Tutorial
RDD (Resilient Distributed Dataset) is a fundamental data structure of Spark and it is the primary data abstraction in Apache Spark and the Spark Core. RDDs are fault-tolerant, immutable distributed collections of objects, which means once you create an RDD you cannot change it. Each dataset in RDD is divided into logical partitions, which can be computed on different nodes of the cluster. 
 
This Apache Spark RDD Tutorial will help you start understanding and using Apache Spark RDD (Resilient Distributed Dataset) with Scala code examples. All RDD examples provided in this tutorial were also tested in our development environment and are available at GitHub spark scala examples project for quick reference.
 
In this section of the Apache Spark tutorial, I will introduce the RDD and explains how to create them and use its transformation and action operations. Here is the full article on Spark RDD in case if you wanted to learn more of and get your fundamentals strong.
 
RDD creation
RDD’s are created primarily in two different ways, first parallelizing an existing collection and secondly referencing a dataset in an external storage system (HDFS, HDFS, S3 and many more). 
 
sparkContext.parallelize()
sparkContext.parallelize is used to parallelize an existing collection in your driver program. This is a basic method to create RDD.
 
!!
//Create RDD from parallelize    
val dataSeq = Seq(("Java", 20000), ("Python", 100000), ("Scala", 3000))   
val rdd=spark.sparkContext.parallelize(dataSeq)
!!
 
Scala
sparkContext.textFile()
Using textFile() method we can read a text (.txt) file from many sources like HDFS, S#, Azure, local e.t.c into RDD.
 
 
 
!!
//Create RDD from external Data source
val rdd2 = spark.sparkContext.textFile("/path/textFile.txt")
!!
 
Scala
RDD Operations
On Spark RDD, you can perform two kinds of operations.
 
RDD Transformations
Spark RDD Transformations are lazy operations meaning they don’t execute until you call an action on RDD. Since RDD’s are immutable, When you run a transformation(for example map()), instead of updating a current RDD, it returns a new RDD.
 
Some transformations on RDD’s are flatMap(), map(), reduceByKey(), filter(), sortByKey() and all these return a new RDD instead of updating the current.
 
RDD Actions
RDD Action operation returns the values from an RDD to a driver node. In other words, any RDD function that returns non RDD[T] is considered as an action. RDD operations trigger the computation and return RDD in a List to the driver program.
 
Some actions on RDD’s are count(),  collect(),  first(),  max(),  reduce()  and more.
 
RDD Examples
How to read multiple text files into RDD
Read CSV file into RDD
Ways to create an RDD
Create an empty RDD
RDD Pair Functions
Generate DataFrame from RDD
Spark DataFrame Tutorial with Basic Examples
DataFrame definition is very well explained by Databricks hence I do not want to define it again and confuse you. Below is the definition I took it from Databricks.
 
DataFrame is a distributed collection of data organized into named columns. It is conceptually equivalent to a table in a relational database or a data frame in R/Python, but with richer optimizations under the hood. DataFrames can be constructed from a wide array of sources such as structured data files, tables in Hive, external databases, or existing RDDs.
 
– Databricks
DataFrame creation
The simplest way to create a DataFrame is from a seq collection. DataFrame can also be created from an RDD and by reading files from several sources.
 
using createDataFrame()
By using createDataFrame() function of the SparkSession you can create a DataFrame.
 
!!
 
val data = Seq(('James','','Smith','1991-04-01','M',3000),
  ('Michael','Rose','','2000-05-19','M',4000),
  ('Robert','','Williams','1978-09-05','M',4000),
  ('Maria','Anne','Jones','1967-12-01','F',4000),
  ('Jen','Mary','Brown','1980-02-17','F',-1)
)
 
val columns = Seq("firstname","middlename","lastname","dob","gender","salary")
df = spark.createDataFrame(data), schema = columns).toDF(columns:_*)
!!
 
Scala
Since DataFrame’s are structure format which contains names and column, we can get the schema of the DataFrame using df.printSchema()
 
df.show() shows the 20 elements from the DataFrame.
 
 No. Group A : Data Science
1. Data Wrangling, I
Perform the following operations using Python on any open source dataset (e.g., data.csv)
1. Import all the required Python Libraries.
2. Locate an open source data from the web (e.g., https://www.kaggle.com). Provide a clear
 description of the data and its source (i.e., URL of the web site).
3. Load the Dataset into pandas dataframe.
4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe()
function to get some initial statistics. Provide variable descriptions. Types of variables etc.
Check the dimensions of the data frame.
5. Data Formatting and Data Normalization: Summarize the types of variables by checking
the data types (i.e., character, numeric, integer, factor, and logical) of the variables in the
data set. If variables are not in the correct data type, apply proper type conversions.
6. Turn categorical variables into quantitative variables in Python.
In addition to the codes and outputs, explain every operation that you do in the above steps and
explain everything that you do to import/read/scrape the data set.
2. Data Wrangling II
Create an “Academic performance” dataset of students and perform the following operations using
Python.
1. Scan all variables for missing values and inconsistencies. If there are missing values and/or
inconsistencies, use any of the suitable techniques to deal with them.
2. Scan all numeric variables for outliers. If there are outliers, use any of the suitable
techniques to deal with them.
3. Apply data transformations on at least one of the variables. The purpose of this
transformation should be one of the following reasons: to change the scale for better
understanding of the variable, to convert a non-linear relation into a linear one, or to
decrease the skewness and convert the distribution into a normal distribution.
Reason and document your approach properly.
k.
Curriculum for Third Year of Computer Engineering (2019 Course), Savitribai Phule Pune University
http://collegecirculars.unipune.ac.in/sites/documents/Syllabus2020/Forms/AllItems.aspx #83/87
3. Descriptive Statistics - Measures of Central Tendency and variability
Perform the following operations on any open source dataset (e.g., data.csv)
1. Provide summary statistics (mean, median, minimum, maximum, standard deviation) for
a dataset (age, income etc.) with numeric variables grouped by one of the qualitative
(categorical) variable. For example, if your categorical variable is age groups and
quantitative variable is income, then provide summary statistics of income grouped by the
age groups. Create a list that contains a numeric value for each response to the categorical
variable.
2. Write a Python program to display some basic statistical details like percentile, mean,
standard deviation etc. of the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’
of iris.csv dataset.
Provide the codes with outputs and explain everything that you do in this step.
l.
4. Data Analytics I
Create a Linear Regression Model using Python/R to predict home prices using Boston Housing
Dataset (https://www.kaggle.com/c/boston-housing). The Boston Housing dataset contains
information about various houses in Boston through different parameters. There are 506 samples
and 14 feature variables in this dataset.
The objective is to predict the value of prices of the house using the given features.
5. Data Analytics II
1. Implement logistic regression using Python/R to perform classification on
Social_Network_Ads.csv dataset.
2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall
on the given dataset.
6. Data Analytics III
1. Implement Simple Naïve Bayes classification algorithm using Python/R on iris.csv dataset.
2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall
on the given dataset.
7. Text Analytics
1. Extract Sample document and apply following document preprocessing methods:
Tokenization, POS Tagging, stop words removal, Stemming and Lemmatization.
2. Create representation of document by calculating Term Frequency and Inverse Document
Frequency.
8. Data Visualization I
1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information
about the passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to
see if we can find any patterns in the data.
2. Write a code to check how the price of the ticket (column name: 'fare') for each passenger
is distributed by plotting a histogram.
9. Data Visualization II
1. Use the inbuilt dataset 'titanic' as used in the above problem. Plot a box plot for distribution
of age with respect to each gender along with the information about whether they survived
or not. (Column names : 'sex' and 'age')
2. Write observations on the inference from the above statistics.
Curriculum for Third Year of Computer Engineering (2019 Course), Savitribai Phule Pune University
http://collegecirculars.unipune.ac.in/sites/documents/Syllabus2020/Forms/AllItems.aspx #84/87
10. Data Visualization III
Download the Iris flower dataset or any other dataset into a DataFrame. (e.g.,
https://archive.ics.uci.edu/ml/datasets/Iris ). Scan the dataset and give the inference as:
1. List down the features and their types (e.g., numeric, nominal) available in the dataset.
2. Create a histogram for each feature in the dataset to illustrate the feature distributions.
3. Create a boxplot for each feature in the dataset.
4. Compare distributions and identify outliers.
Group B- Big Data Analytics – JAVA/SCALA (Any three)
1. Write a code in JAVA for a simple WordCount application that counts the number of
occurrences of each word in a given input set using the Hadoop MapReduce framework on
local-standalone set-up.
2. Design a distributed application using MapReduce which processes a log file of a system.
3. Locate dataset (e.g., sample_weather.txt) for working on weather data which reads the text
input files and finds average for temperature, dew point and wind speed.
4. Write a simple program in SCALA using Apache Spark framework
 

 

 
 
 
 
 
 
 
 
 
 
 
 
 

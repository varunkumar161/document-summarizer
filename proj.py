
from nltk import *
import csv
import os 
import math
import numpy as np
from nltk.corpus import stopwords
import nltk.data
import numpy as np
np.set_printoptions(threshold=np.inf)
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
import pandas
from collections import Counter
from scipy.cluster.vq import kmeans2
from sklearn.preprocessing import normalize
from nltk.tokenize import sent_tokenize, word_tokenize

txt_list=[]
current_path="D:/tfpy"
for file in os.listdir(current_path+"/documents"):
    if file.endswith(".txt"):
        txt_list.append(file)

txt_content=""
for file in txt_list:
    txt_file_path=current_path+"/documents/"+file
    with open(txt_file_path,'r') as f:
        content=f.read().lower()
        txt_content+=content
# tokenising followed by pos tagging
#print txt_content
text=word_tokenize(txt_content)

ele = ['?', '!', ':', ';', '(', ')', '[', ']', '{', '}','*','#','$','/','`','"','.','%','--','__','``','.', ',', '"', "'",]

stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])



filtered_words = [word for word in text if word not in stop_words]
#print stop_words


for word in filtered_words:
    if word in ele:
            filtered_words.remove(word)


seen = set()
result = []
for word in filtered_words:
    if word not in seen:
        seen.add(word)
        result.append(word)
        
#print text
        filtered_words = result
        #print result
        
with open(current_path+"/after_preprocess.txt",'w') as f:
    txt_post_tagged=pos_tag(filtered_words)
    list_of_words=[word[0] for word in txt_post_tagged] #this words list is calculat
    str_copy_txt_pos_tagged=str(txt_post_tagged)
    f.write(str_copy_txt_pos_tagged)
    print filtered_words

    
def tf (word,document_name):
	with open ("D:/tfpy/documents/"+document_name,'r') as f:
		content=f.read().lower()

	no_of_occurences=content.count(word)
	total_no_of_words_in_the_document=float(len(content.split()))
	value=no_of_occurences/total_no_of_words_in_the_document
	

	return value
def idf (word,doc_list):
	current_path="D:/tfpy/documents/"
	no_of_documents_the_word_occured=0
	for file in doc_list:
		with open(current_path+file,'r') as f:
			content=f.read().lower()
			if (content.count(word)) > 0:
				no_of_documents_the_word_occured+=1
	
	#print str(no_of_documents_the_word_occured) +"  "+word

	if no_of_documents_the_word_occured==0:
		value= 0
	else:
		value= math.log((float(len(doc_list)))/(float(no_of_documents_the_word_occured)))

	return value

def tfidf (word,document_name,doc_list):
	return tf(word,document_name)*idf(word,doc_list)

# making the tfidf matrix
rows_of_matrix=len(list_of_words)
columns_of_matrix=len(txt_list)

tfidf_matrix=np.zeros((rows_of_matrix,columns_of_matrix), dtype=np.float)

for i in range(0,rows_of_matrix):
	for j in range(0,columns_of_matrix):
		word=list_of_words[i]
		document_name=txt_list[j]
		doc_list=txt_list
		tfidf_matrix[i][j]=tfidf(word,document_name,txt_list)

B = np.array(tfidf_matrix)
#column_labels = len(txt_list)
#print mat(B,row_labels=row_labels, col_labels=column_labels)
df = pandas.DataFrame(B, index=list_of_words, columns=txt_list)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
print df
pandas.set_option('display.height', 10000)
pandas.set_option('display.width', 10000)

def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''

    n, m = A.shape
    x = randomUnitVector(m)
    lastV = None
    currentV = x

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV


def svd(A, k=None, epsilon=1e-10):
    '''
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    '''
    n, m = A.shape
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
        u_unnormalized = np.dot(A, v)
        sigma = norm(u_unnormalized)  # next singular value
        u = u_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return singularValues, us.T, vs

if __name__ == "__main__":

    # v1 = svd_1d(movieRatings)
    # print(v1)
    A = np.array(tfidf_matrix)
    matrix = A
    #matrix = normalize(matrix)
    matrix = normalize(matrix, norm='l1', axis=1)
    sigma, U, V = svd(matrix)
    
    data = sigma, U, V
    print sigma
    print "singular values ####"
    """
    indexToWord= list_of_words
    indexToDocument = txt_list
    projectedDocuments = np.dot(matrix.T, U)
    projectedWords = np.dot(matrix, V.T)
    #documentCenters, documentClustering = cluster(projectedDocuments)
    wordCenters, wordClustering = cluster(projectedWords)
    wordClusters = [
        [indexToWord[i] for (i, x) in enumerate(wordClustering) if x == j]
        for j in range(len(set(wordClustering)))
    ] 
    """
    #documentClusters = [
        #[indexToDocument[i]for (i, x) in enumerate(documentClustering) if x == j]for j in range(len(set(documentClustering)))]
    """
print wordClusters
"""
"""
bf = pandas.DataFrame(U, index=list_of_words, columns=txt_list)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
print bf
pandas.set_option('display.height', 10000)
pandas.set_option('display.width', 10000)
"""

m = np.matrix(U)
a = m.max()
b = m.min()
c= (a+b)/2
d= U.shape
s=d[0]
t=d[1]
print s
print t
print c, a, b
ls=[]
for i in range(s):
 for j in range(t):
   if  U[i][j] >=c:
    ls.append(list_of_words[i])

seen = set()
result = []
for word in ls:
    if word not in seen:
        seen.add(word)
        result.append(word)
    
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
current_path="D:/tfpy/documents/"
crnt_path="D:/tfpy/summaries/"
ls = result
#print ls
for i in range(len(doc_list)):
    file_name= doc_list[i]
    with open(current_path+file_name,'r') as g:
        conn=g.read().lower()
        conn=sent_tokenize(conn)
        target = open(crnt_path+file_name,'w')
        sent=set()
        for word in ls:
            for sentence in conn:
                if word in sentence:
                    sent.add(sentence)


        connt= str(sent)
        target.write(connt)                   
target.close()
g.close()

                    

    

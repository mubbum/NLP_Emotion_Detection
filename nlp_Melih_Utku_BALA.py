### import of libraries ### 
import pandas as pd
import re
import snowballstemmer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score,accuracy_score



# Removal of numeric characters
def remove_numeric(value):
    bfr = [item for item in value if not item.isdigit()]
    bfr = "".join(bfr)
    return bfr 

# Removal of emojis
def remove_emoji(value):
    bfr = re.compile("[\U00010000-\U0010ffff]", flags = re.UNICODE)
    bfr = bfr.sub(r'',value)
    return bfr

# Removal of punctuation
def remove_noktalama(value):
    return re.sub(r'[^\w\s]','',value) 

# Removal of single-character expressions
def remove_single_chracter(value):
    return re.sub(r'(?:^| )\w(?:$| )','',value)

# Removal of links
def remove_link(value):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',value)

# Removal of hashtags
def remove_hashtag(value):
    return re.sub(r'#[^\s]+','',value)

# Removal of usernames
def remove_username(value):
    return re.sub(r'@[^\s]+','',value)


# Stem reduction and stop words operations
def stem_word(value):
    stemmer = snowballstemmer.stemmer('turkish')
    value = value.lower()
    value = stemmer.stemWords(value.split())
    stop_words = ["acaba","ama","aslında","az","bazı","belki","biri","birkaç","birşey","biz","bu","çok",
                 "çünkü","da","de","daha","defa","diye","eğer","en","gibi","hep","hem","hepsi","her",
                 "hiç","için","ile","ise","kez","ki","kim","mi","mı","mu","mü","nasıl","ne","neden","nerde",
                 "nerede","nereye","niçin","niye","o","sanki","şey","siz","şu","tüm","ve","veya","ya","yani"
                 ,"bir","iki","üç","dört","beş","altı","yedi","sekiz","dokuz","on"]
    value = [item for item in value if not item in stop_words]
    value = "".join(value)
    return value

# Calling preprocessing functions sequentially
def pre_processing(value):
    return [remove_numeric(remove_emoji
                          (remove_single_chracter
                           (remove_noktalama
                            (remove_link
                             (remove_hashtag
                              (remove_username
                               (stem_word(word)))))))) for word in value.split()]

# Removal of spaces
def remove_space(value):
    return [item for item in value if item.strip()]

# Creating and saving a word2vec model
def word2vec_create(value):
    model = Word2Vec(sentences = value.tolist(),vector_size=100,window=5,min_count=1)
    model.save("data/word2vec.model")

# word2vec model loading and vector extraction
def word2vec(value):
    model = Word2Vec.load("data/word2vec.model")
    bfr_list = []
    bfr_len = len(value)

    for k in value:
        bfr = model.wv.key_to_index[k]
        bfr = model.wv[bfr]
        bfr_list.append(bfr)

    bfr_list = sum(bfr_list)
    bfr_list = bfr_list/bfr_len
    return bfr_list.tolist()

# word2vec model is updated
def word2vec_update(value):
    model = Word2Vec.load("data/word2vec.model")
    model.build_vocab(value.tolist(),update=True)
    model.save("data/word2vec.model")


if __name__ == '__main__':
   
    # Our sample dataset is read for data cleaning
    df_1 = pd.read_csv("data/nlp.csv",index_col=0)


    ### Create a new column in df_1 as Text_2 by passing the Text column in the defined df_1 through the preprocessing functions. ###
    df_1["Text_2"] = df_1["Text"].apply(pre_processing)
    df_1["Text_2"] = df_1["Text_2"].apply(remove_space)

    ### Empty list control in Text_2 column in df_1 ###
    df_1[df_1["Text_2"].str[0].isnull()]

    df_index = df_1[df_1["Text_2"].str[0].isnull()].index
    df_1 = df_1.drop(df_index)
    df_1 = df_1.reset_index()
    del df_1["index"]

    df_1[df_1["Text_2"].str[0].isnull()]
    
    ### word2vec model creation ###
    word2vec_create(df_1["Text_2"])
    df_1["word2vec"] = df_1["Text_2"].apply(word2vec)
    
    # Since we will not be using our df_1 dataframe anymore, we assign an empty value so that it does not take up space in RAM.
    df_1 = {}

    #############################################################################################################################################

    # The data we will classify is read
    df_2 = pd.read_csv("data/metin_siniflandirma.csv",index_col=0)

    ### Create a new column in df_2 as Text_2 by passing the Text column in the defined df_2 through the preprocessing functions. ###
    df_2["Text_2"] = df_2["Text"].apply(pre_processing)
    df_2["Text_2"] = df_2["Text_2"].apply(remove_space)
    
    ### Empty list control in Text_2 column in df_2 ###
    df_2[df_2["Text_2"].str[0].isnull()]

    df_index = df_2[df_2["Text_2"].str[0].isnull()].index
    df_2 = df_2.drop(df_index)
    df_2 = df_2.reset_index()
    del df_2["index"]

    df_2[df_2["Text_2"].str[0].isnull()]

    ### Update the Text_2 column data in the df_2 that we will classify in the word2vec data. ### 
    word2vec_update(df_2["Text_2"])

    ### Create a new column using this model called word2vec over text_2 columns
    df_2["word2vec"] = df_2["Text_2"].apply(word2vec)

    ### Split word2vec column as train test ###
    msg_train,msg_test,label_train,label_test = train_test_split(df_2["word2vec"].tolist(),df_2["Label"].tolist(),test_size=0.2,random_state=42)

    ### Create svm pipeline, train and test the model ###
    svm = Pipeline([("svm",LinearSVC())])
    svm.fit(msg_train,label_train)
    y_pred_class = svm.predict(msg_test)
    
    ### Show accuracy and f1 score output with print ###
    print("svm accuary score :", accuracy_score(label_test,y_pred_class))
    print("svm f1 score :", f1_score(label_test,y_pred_class,average="weighted"))

    
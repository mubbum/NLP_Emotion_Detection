### kütüphaneleri tanımlayınız. ### 
import pandas as pd
import re
import snowballstemmer
#from gensim.models import Word2Vec

df = pd.read_csv("data/nlp.csv")

### tanımlanan fonksiyonlar da pass'lı ifadeler eksiktir. Bu fonksiyon içeriklerini doldurunuz ###

# numerik karakterlerin kaldırılması
def remove_numeric(value):
    bfr = [item for item in value if not item.isdigit()]
    bfr = "".join(bfr)
    return bfr 

# emojilerin kaldırılması
def remove_emoji(value):
    bfr = re.compile("[\U00010000-\U0010ffff]", flags = re.UNICODE)
    bfr = bfr.sub(r'',value)
    return bfr

#noktalama işaretlerinin kaldırılması
def remove_noktalama(value):
    return re.sub(r'[^\w\s]','',value) 

#tek karakterli ifadelerin kaldırılması
def remove_single_chracter(value):
    return re.sub(r'(?:^| )\w(?:$| )','',value)

#linklerin kaldırılması 
def remove_link(value):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',value)

# hashtaglerin kaldırılması
def remove_hashtag(value):
    return re.sub(r'#[^\s]+','',value)

# kullanıcı adlarının kaldırılması
def remove_username(value):
    return re.sub(r'@[^\s]+','',value)


#kök indirgeme ve stop words işlemleri
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

# ön işlem fonksiyonlarının sırayla çağırılması
def pre_processing(value):
    return [remove_numeric(remove_emoji
                          (remove_single_chracter
                           (remove_noktalama
                            (remove_link
                             (remove_hashtag
                              (remove_username
                               (stem_word(word)))))))) for word in value.split()]

# Boşlukların kaldırılması
def remove_space(value):
    return [item for item in value if itrm.strip()]

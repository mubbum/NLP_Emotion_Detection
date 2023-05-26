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
    print(bfr)
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
    return [remove_numeric(remove_numeric
                          (remove_single_chracter
                           (remove_noktalama
                            (remove_link
                             (remove_hashtag
                              (remove_username
                               (stem_word(word)))))))) for word in value.split()]

df["Text"].apply(pre_processing)
# Boşlukların kaldırılması
def remove_space(value):
    pass

# word2vec model oluşturma ve kaydetme
def word2vec_create(value):
    model = Word2Vec(sentences = value.tolist(),vector_size=100,window=5,min_count=1)
    model.save("data/word2vec.model")

# word2vec model yükleme ve vektör çıkarma
def word2vec_analysis(value):
    model = Word2Vec.load("data/word2vec.model")
    pass

# word2vec model güncellenir.
def word2vec_update(value):
    model = Word2Vec.load("data/word2vec.model")
    model.build_vocab(value.tolist(),update=True)
    model.save("data/word2vec.model")


if __name__ == '__main__':
   
    # veri temizlemesi için örnek veri kümemiz okunur.
    df_1 = pd.read_csv("data/nlp.csv",index_col=0)


    ### tanımlanan df_1 içerisinde Text sütununu ön işlem fonksiyonlarından geçirerek Text_2 olarak df_1 içerisinde yeni bir sütun oluşturun. ###
    

    ### df_1 içerisinde Text_2 sütununda boş liste kontrolü ###

    
    ### word2vec model oluşturma ###
    
    
    # df_1 dataframe mizi artık kullanmaycağımızdan ram de yer kaplamaması adına boş bir değer ataması yapıyoruz.
    df_1 = {}

    #############################################################################################################################################

    # sınıflandırma yapacağımız veri okunur.
    df_2 = pd.read_csv("data/metin_siniflandirma.csv",index_col=0)

    ### tanımlanan df_2 içerisinde Text sütununu ön işlem fonksiyonlarından geçirerek Text_2 olarak df_2 içerisinde yeni bir sütun oluşturun. ###

    
    ### df_2 içerisinde Text_2 sütununda boş liste kontrolü ###


    ### sınıflandırma yapacağımız df_2 içerisinde bulunan Text_2 sütun verisini word2vec verisinde güncelleyin. ### 


    ### Text_2 sütun üzerinden word2vec adında bu modeli kullanarak yeni bir sütun yaratın


    ### word2vec sütunumuzu train test olarak bölün ###


    ### svm pipeline oluştur, modeği eğit ve test et ###

    
    ### accuracy ve f1 score çıktısını print ile gösterin. ###

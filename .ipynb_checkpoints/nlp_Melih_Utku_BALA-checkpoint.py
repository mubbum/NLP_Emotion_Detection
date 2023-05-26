### kütüphaneleri tanımlayınız. ### 
import pandas as pd
#from gensim.models import Word2Vec


### tanımlanan fonksiyonlar da pass'lı ifadeler eksiktir. Bu fonksiyon içeriklerini doldurunuz ###

# numerik karakterlerin kaldırılması
def remove_numeric(value):
    pass 

# emojilerin kaldırılması
def remove_emoji(value):
    pass

#noktalama işaretlerinin kaldırılması
def remove_noktalama(value):
    pass 

#tek karakterli ifadelerin kaldırılması
def remove_single_chracter(value):
    pass

#linklerin kaldırılması 
def remove_link(value):
    pass

# hashtaglerin kaldırılması
def remove_hashtag(value):
    pass

# kullanıcı adlarının kaldırılması
def remove_username(value):
    pass


#kök indirgeme ve stop words işlemleri
def stem_word(value):
    pass

# ön işlem fonksiyonlarının sırayla çağırılması
def pre_processing(value):
    pass

# Boşlukların kaldırılması
def remove_space(value):
    pass

# word2vec model oluşturma ve kaydetme
#def word2vec_create(value):
 #   model = Word2Vec(sentences = value.tolist(),vector_size=100,window=5,min_count=1)
  #  model.save("data/word2vec.model")

# word2vec model yükleme ve vektör çıkarma
#def word2vec_analysis(value):
 #   model = Word2Vec.load("data/word2vec.model")
  #  pass

# word2vec model güncellenir.
#def word2vec_update(value):
 #   model = Word2Vec.load("data/word2vec.model")
  #  model.build_vocab(value.tolist(),update=True)
   # model.save("data/word2vec.model")


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

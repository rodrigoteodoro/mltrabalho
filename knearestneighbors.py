# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import scipy.stats
from scipy.spatial.distance import hamming
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection  import train_test_split
import itertools
from sklearn import metrics
import seaborn as sns
import gc
import warnings

# k-nearest neighbors

products_df = pd.read_csv("./instacart/products.csv")

# Cria uma função que retorna os dados do produto por id.
def productsMeta(productId):
    products_df.columns
    product_id = products_df.at[productId,"product_id"]
    product_name = products_df.at[productId,"product_name"]
    return product_id, product_name

# Carrega o dataset - No dataset part não temos produtos comprados mais de uma vez por usuário
orders_prior_train_df = pd.read_csv('orders_prior_train_df_maior.csv') 
orders_prior_train_df.shape
orders_prior_train_df.head(1)

orders_resume = orders_prior_train_df.groupby(['user_id', 'product_id']).size().reset_index(name='rating')
orders_resume_shapeInicial = orders_resume.shape
orders_resume.shape
orders_resume.head(5)
orders_resume.columns
# del orders_prior_train_df

n_users = orders_resume.user_id.unique().shape[0]
n_items = orders_resume.product_id.unique().shape[0]
print('users %s' % n_users)
print('items %s' % n_items)

# Verifica se tem usários que compraram mais de um produto nos registros
def check_userid_produtos():
    df = orders_resume.query('rating > 1')
    print(df.head(10))
    return df

# Usuários que compraram mais de um produto de uma vez
dfUsuariosProdutosMaisVezes = check_userid_produtos()
dfUsuariosProdutosMaisVezes.shape
dfUsuariosProdutosMaisVezes = dfUsuariosProdutosMaisVezes.sort_values(["rating"], ascending=False)
dfUsuariosProdutosMaisVezes.head(10)

# Separa o N usuarios que mais compraram produtos 
orders_resume = orders_resume[orders_resume['user_id'].isin(dfUsuariosProdutosMaisVezes[:1000].user_id.unique().tolist())]
n_users = orders_resume.user_id.unique().shape[0]
n_items = orders_resume.product_id.unique().shape[0]
print('users %s' % n_users)
print('items %s' % n_items)


# Quantidade de produtos por usuário
USERPorProdutos = orders_resume.product_id.value_counts()
USERPorProdutos.head()

# Quantidade de usuários que compraram pelo menos um produto
PRODUTOSPorUser = orders_resume.user_id.value_counts()
PRODUTOSPorUser.head()

# Deixa da base apenas produtos comprados por mais de 10 usuários
orders_resume = orders_resume[orders_resume["product_id"].isin(USERPorProdutos[USERPorProdutos>10].index)]

# Deixa na base apenas usuários que tenham comprado mais de 10 produtos
orders_resume = orders_resume[orders_resume["user_id"].isin(PRODUTOSPorUser[PRODUTOSPorUser>10].index)]

# Verifica novamente o shape
orders_resume_shapeFinal = orders_resume.shape

# Compara o quando foi removido
print("orders_resume_shapeInicial: ", orders_resume_shapeInicial)
print("orders_resume_shapeFinal: ", orders_resume_shapeFinal)
print('users %s' % orders_resume.user_id.unique().shape[0])
print('items %s' % orders_resume.product_id.unique().shape[0])

# Verifica se tem usários que compraram mais de um produto nos registros
check_userid_produtos()

# Cria uma matriz produtos x usuários
userItemRatingMatrix=pd.pivot_table(orders_resume, values='rating', index=['user_id'], columns=['product_id'])
userItemRatingMatrix.head()
userItemRatingMatrix.shape

# Calcula a distância hamming entre os vetores de classificações
# de dois usuários.
def distance(user1, user2):
    try:
        user1Ratings = userItemRatingMatrix.transpose()[user1]
        user2Ratings = userItemRatingMatrix.transpose()[user2]
        distance = hamming(user1Ratings,user2Ratings)
    except:
            distance = np.NaN
    return distance

# Função que retornar os 10 usuários mais semelhantes a dado usuário
def nearestNeighbors(user, K=10):
    allUsers = pd.DataFrame(userItemRatingMatrix.index)
    allUsers = allUsers[allUsers.user_id!=user]
    allUsers["distance"] = allUsers["user_id"].apply(lambda x: distance(user,x))
    KnearestUsers = allUsers.sort_values(["distance"],ascending=True)["user_id"][:K]
    return KnearestUsers

# Esta função gera uma lista de recomendações de produtos não comprados
# dado usuário de acordo com produto comprados por outros usuários semelhantes
def topN(user, N=3):
    #user = 210
    KnearestUsers = nearestNeighbors(user)
    NNRatings = userItemRatingMatrix[userItemRatingMatrix.index.isin(KnearestUsers)]    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # avgRating = NNRatings.apply(np.nanmean).dropna()
        avgRating = NNRatings.apply(lambda x:scipy.stats.mode(x)[0][0]).dropna()
        #avgRating.head()
                
    produtosRealmenteComprados = userItemRatingMatrix.transpose()[user].dropna().index
    # Quantidade média de produtos comprados no período pelos os demais usuários
    avgRating = avgRating[~avgRating.index.isin(produtosRealmenteComprados)]
    ratingPredictedValue = avgRating.sort_values(ascending=False)
    ratingPredictedValue.head(20)
    # print(ratingPredictedValue.head(N))
    topNProductIDs = avgRating.sort_values(ascending=False).index[:N]
    recommendation = pd.DataFrame(topNProductIDs)
    recommendation["product"] = recommendation["product_id"].apply(productsMeta)    
    recommendation["Prediction"] = ratingPredictedValue.values[:N]
    return recommendation

def recomenacoes(userid, N=10):
    print('Usuário: %s' % userid)
    recommendation = topN(userid, N)
    # recommendation.head(10)    
    recommendation.sort_values('Prediction', ascending=False)
    print(recommendation.head(N))
    print('------------------------------------------------------------------')

# Recomendções para um usuário específico
recomenacoes(90)
'''
Usuário: 90
   product_id                                            product  Prediction
0       21386                    (21387, Non-Fat Vanilla Yogurt)        51.0
1       49683  (49684, Vodka, Triple Distilled, Twist of Vani...        49.0
2        1463                   (1464, Special k Protein Cereal)        37.0
3       10960                    (10961, Five Cheese Tortellini)        33.0
4       45437                (45438, Select 2% Reduced Fat Milk)        32.0
5       26317                           (26318, Pineapple Juice)        31.0
6        5876             (5877, Organic Rose Maqui & Cocao Bar)        30.0
7        1940          (1941, Sea Salt And Vinegar Roasted Corn)        26.0
8       47156              (47157, Espresso Roast Ground Coffee)        25.0
9       18811                       (18812, Vegan Smoked Salmon)        25.0
------------------------------------------------------------------
'''

# Para cada usuário 10 recomendações
limite = 1
for user_id in orders_resume.user_id.unique():
    recomenacoes(user_id)
    limite -=1
    if limite < 1:
        break



# Remove itens da memória
gc.collect()




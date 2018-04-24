# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import scipy.stats
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

# K-means

products_df = pd.read_csv("./instacart/products.csv")
aisles_df = pd.read_csv("./instacart/aisles.csv")


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


# Ratings
ratings = pd.pivot_table(orders_resume, values='rating', index=['user_id'], columns=['product_id'], fill_value=0)
ratings.shape
ratings.head(10)
ratings.columns
#writer = pd.ExcelWriter('ratings.xlsx')
#ratings[:100].to_excel(writer,'Sheet1', index=False)
#writer.save()


# Aplica o Kmeans
n_components=10
pca = PCA(n_components=n_components, random_state=0)
pca.fit(ratings)
pca_samples = pca.transform(ratings)
print(pca_samples)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)  

ps = pd.DataFrame(pca_samples)
ps.head()

tocluster = pd.DataFrame(ps[[(n_components-1), 1]])
tocluster.shape
tocluster.head()

ncluster = n_components-1  # Deve ser menor que número de n_components
tocluster.columns

fig = plt.figure(figsize=(12,8))
plt.plot(tocluster[ncluster], tocluster[1], 'o', markersize=2, color='blue', alpha=0.5, label='class')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()


clusterer = KMeans(init='k-means++', n_clusters=ncluster, random_state=0, n_init=1).fit(tocluster)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(tocluster)

print(centers)
#c_preds[0:100]

fig = plt.figure(figsize=(12, 8))
colors = sns.color_palette("Set2", ncluster)
colored = [colors[k] for k in c_preds]
plt.scatter(tocluster[ncluster], tocluster[1],  color = colored)
for ci,c in enumerate(centers):
    plt.plot(c[0], c[1], 'o', markersize=8, color='red', alpha=0.9, label=''+str(ci))
plt.xlabel('x_values')
plt.ylabel('y_values')
#plt.legend()
plt.show()

clust_prod = ratings.copy()
clust_prod['cluster'] = c_preds
clust_prod.head(10)
clust_prod.shape
# Salvaer em excel para ver

# Separação dos usuarios clusterizados
lst_cluster = []

for idx in range(0, ncluster):
    # print(idx)
    c0 = clust_prod[clust_prod['cluster']==idx]
    c0.shape
    lst_cluster.append(c0)
    #c0.head(10)

# Regata os usuários do grupo e realiza a recomendação
ratings2 = orders_resume.pivot_table(index='user_id', columns='product_id', values='rating')

# Localiza um usuário nos clusters
def localizarCluster(userid):
    r = -1
    for idx in range(0, ncluster):
        if userid  in lst_cluster[idx].drop('cluster',axis=1).reset_index('user_id').user_id.unique():
            r = idx
            break
    return r

def topN(user, N=3):
    idx = localizarCluster(user) # Clustar    
    NNRatings = ratings2[ratings2.index.isin(lst_cluster[idx].drop('cluster',axis=1).reset_index('user_id').user_id.unique())]
    NNRatings.head()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        #avgRating = NNRatings.apply(np.nanmean).dropna()
        avgRating = NNRatings.apply(lambda x:scipy.stats.mode(x)[0][0]).dropna()
        # avgRating.head(10)

    produtosRealmenteComprados = ratings2.transpose()[user].dropna().index
    # produtosRealmenteComprados
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

def recomenacoes(userid, N=10, comparar_produtos=[]):
    print('Usuário: %s' % userid)
    recommendation = topN(userid, N)
    recommendation.head(N)
    recommendation.sort_values('Prediction', ascending=False)
    if len(comparar_produtos) == 0:
        print(recommendation.head(N))
        print('------------------------------------------------------------------')
    if comparar_produtos:
        print('Comparando produtos')
    for i, row in recommendation.iterrows():
        # print(row['product'][0])
        if row['product'][0] in comparar_produtos:
            print(row['product'], row['Prediction'])
    print('------------------------------------------------------------------')
    return recommendation

for userid in orders_resume.user_id.unique()[:10]:
    recomenacoes(userid, N=10)

recomenacoes(90, N=10)

# Resgata o corredor dos produtos
recommendation = recomenacoes(90, N=50)
product_ids = []
for i, row in recommendation.iterrows():
    product_ids.append(row['product'][0])

products_df2 = pd.merge(products_df, aisles_df, on='aisle_id', how='left')
products_df2[['product_id', 'product_name', 'aisle']][products_df2['product_id'].isin(product_ids)].sort_values('product_name')

# Remove itens da memória
gc.collect()



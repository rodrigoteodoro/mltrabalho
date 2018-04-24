#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:32:11 2018

@author: root
"""

import numpy as np 
import pandas as pd
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import scipy.stats
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection  import train_test_split
from sklearn.neighbors import NearestNeighbors
import itertools
from sklearn import metrics
from time import time
import gc
from orangecontrib.associate.fpgrowth import *
import gc

color = sns.color_palette()

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = 50
pd.set_option('display.width', 200)

#  FP-Growth

products_df = pd.read_csv("./instacart/products.csv")
products_df.shape

aisles_df = pd.read_csv("./instacart/aisles.csv")
aisles_df.shape

departments_df = pd.read_csv("./instacart/departments.csv")
departments_df.shape

# Data set de trabalho - estatística descritiva -------------------------------
orders_prior_train_df = pd.read_csv('orders_prior_train_df_maior.csv') 
orders_prior_train_df.shape
orders_prior_train_df.head(10)

# Produtos completo
products_completo_df = pd.merge(products_df, aisles_df, on ='aisle_id',  how='left')
products_completo_df = pd.merge(products_completo_df, departments_df, on='department_id',  how='left')
products_completo_df.head()

# Junta com produtos completos
orders_prior_train_df = pd.merge(orders_prior_train_df, products_completo_df, on ='product_id',  how='left')
    orders_prior_train_df.columns
n_users = orders_prior_train_df.user_id.unique().shape[0]
n_items = orders_prior_train_df.product_id.unique().shape[0]
print('users %s' % n_users)
print('items %s' % n_items)

# Quantidade de produtos por usuário
USERPorProdutos = orders_prior_train_df.product_id.value_counts()
USERPorProdutos.head()

# Quantidade de usuários que compraram pelo menos um produto
PRODUTOSPorUser = orders_prior_train_df.user_id.value_counts()
PRODUTOSPorUser.head()

# Deixa da base apenas produtos comprados por mais de 10 usuários
orders_prior_train_df = orders_prior_train_df[orders_prior_train_df["product_id"].isin(USERPorProdutos[USERPorProdutos>10].index)]

# Deixa na base apenas usuários que tenham comprado mais de 10 produtos
orders_prior_train_df = orders_prior_train_df[orders_prior_train_df["user_id"].isin(PRODUTOSPorUser[PRODUTOSPorUser>10].index)]

n_users = orders_prior_train_df.user_id.unique().shape[0]
n_items = orders_prior_train_df.product_id.unique().shape[0]
print('users %s' % n_users)
print('items %s' % n_items)


# Separar novamente os datasets de validação e treino : 80% de treino 20% de teste
X_train, X_test, y_train, y_test = train_test_split(orders_prior_train_df, orders_prior_train_df.index, test_size=0.20, random_state=0)
print('X_train', X_train.shape) # Dados do dataframe com 80%
print('y_train', y_train.shape) # Indices para Train
print('X_test', X_test.shape) # Dados do dataframe com 20%
print('y_test', y_test.shape) # Indices para Test


# Agrupar Registros por produto
orders_prior_train_df.shape
product_list = orders_prior_train_df.groupby(["order_id"], as_index=True).agg({'product_id': lambda x: list(x)}).reset_index()
product_list.head()
product_list.shape


def f_fpgrowth(product_list, campo_chave, suporte=0.01):
    X = []
    for v in product_list[campo_chave]:
        if len(v) > 1:
            X.append(v)
    # print(len(X))
    # products_df[products_df['product_id'].isin(X[:1][0])]

    itemsets = dict(frequent_itemsets(X, suporte))
    print('Quantidade de grupos totais: %s' % len(itemsets))
    # list(itemsets)

    #  Produtos comprados com frequência juntos
    lista_recomendacao = []
    for trans in itemsets:
        i = []
        for item in trans:
            i.append(item)
        if len(i) > 1:
            l = ''
            if campo_chave == 'product_id':
                p = products_df[products_df[campo_chave].isin(i)]
                for i_, j_ in p.iterrows():
                    l += '%s-%s / ' % (j_[campo_chave], j_['product_name'])
                lista_recomendacao.append(l)
            elif campo_chave == 'aisle_id':
                p = aisles_df[aisles_df[campo_chave].isin(i)]
                for i_, j_ in p.iterrows():
                    l += '%s-%s / ' % (j_[campo_chave], j_['aisle'])
                lista_recomendacao.append(l)
    
    for r in lista_recomendacao:
        print(r)
    print('Quantidade com pelo menos 2 itens: %s' % len(lista_recomendacao))

# Por produtos comprados
f_fpgrowth(product_list, 'product_id')

product_list_aisle_id = orders_prior_train_df.groupby(["order_id"], as_index=True).agg({'aisle_id': lambda x: list(x)}).reset_index()

# Por Corredor de produtos
f_fpgrowth(product_list_aisle_id, 'aisle_id', 0.1)

# Remove itens da memória
gc.collect()

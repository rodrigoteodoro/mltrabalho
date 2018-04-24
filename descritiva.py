#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 09:20:48 2018

@author: root
"""

import numpy as np 
import pandas as pd
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import scipy.stats
from itertools import combinations
from sklearn import metrics
import seaborn as sns
import gc

color = sns.color_palette()

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = 50
pd.set_option('display.width', 200)
# pd.options.display.float_format = '{:.2f}'.format

orders_df = pd.read_csv("./instacart/orders.csv")
orders_df.shape

products_df = pd.read_csv("./instacart/products.csv")
products_df.shape

aisles_df = pd.read_csv("./instacart/aisles.csv")
aisles_df.shape

departments_df = pd.read_csv("./instacart/departments.csv")
departments_df.shape

order_products_train_df = pd.read_csv("./instacart/order_products__train.csv")
order_products_train_df.shape

order_products_prior_df = pd.read_csv("./instacart/order_products__prior.csv")
order_products_prior_df.shape

# Quantidade de registros de cada tipo no dataset de compras ------------------
orders_ct = orders_df.eval_set.value_counts()
plt.figure(figsize=(12,8))
sns.barplot(orders_ct.index, orders_ct.values, alpha=0.8, color=color[1])
plt.ylabel('Número de ocorrências', fontsize=12)
plt.xlabel('Eval_set', fontsize=12)
plt.title('Número de registros em cada dataset', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# Quantidade de produtos ------------------------------------------------------
products_df.shape
products_df.columns
products_df = pd.merge(products_df, aisles_df, on='aisle_id', how='left')
products_df = pd.merge(products_df, departments_df, on='department_id', how='left')

# Por Correddor
product_ct = products_df.groupby('aisle', as_index=False).product_id.count()
product_ct = product_ct.sort_values('product_id', ascending=False)
plt.figure(figsize=(12,8))
sns.set(font_scale=.8)
sns.barplot(product_ct.aisle, product_ct.product_id, alpha=0.8, color=color[1])
plt.ylabel('Número de ocorrências', fontsize=12)
plt.xlabel('Corredor', fontsize=12)
plt.title('Número de registros de produtos por corredor', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# Por departamento
product_ct = products_df.groupby('department', as_index=False).product_id.count()
product_ct = product_ct.sort_values('product_id', ascending=False)
plt.figure(figsize=(12,8))
sns.set(font_scale=.8)
sns.barplot(product_ct.department, product_ct.product_id, alpha=0.8, color=color[1])
plt.ylabel('Número de ocorrências', fontsize=12)
plt.xlabel('Departamento', fontsize=12)
plt.title('Número de registros de produtos por departamento', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

# Juntar Prior com o train ----------------------------------------------------
orders_df.shape
orders_df.columns

order_products_train_df.shape
order_products_train_df.columns

order_products_prior_df.shape
order_products_prior_df.columns

orders_prior_sel_df = orders_df[orders_df['eval_set'].isin(['prior'])]
orders_prior_sel_df.shape
orders_train_sel_df = orders_df[orders_df['eval_set'].isin(['train'])]
orders_train_sel_df.shape

orders_train_sel_df = pd.merge(orders_train_sel_df, order_products_train_df, on='order_id', how='left')
orders_train_sel_df.shape
orders_train_sel_df.columns

orders_prior_sel_df = pd.merge(orders_prior_sel_df, order_products_prior_df, on='order_id', how='left')
orders_prior_sel_df.shape
orders_prior_sel_df.columns

orders_prior_train_df = pd.concat([orders_train_sel_df, orders_prior_sel_df])
orders_prior_train_df.shape
orders_prior_train_df.columns
orders_prior_train_ct = orders_prior_train_df.eval_set.value_counts()
orders_prior_train_ct

# Salva o dataset completo em formato Excel -----------------------------------
orders_prior_train_df.to_csv('orders_prior_train_df.csv', index=False)
orders_prior_train_df[:5000000].to_csv('orders_prior_train_df_maior.csv', index=False)


# Data set de trabalho - estatística descritiva -------------------------------
orders_prior_train_df = pd.read_csv('orders_prior_train_df_maior.csv') 
orders_prior_train_df.shape
orders_prior_train_df.head(10)

orders_ct = orders_prior_train_df.eval_set.value_counts()
plt.figure(figsize=(12,8))
sns.barplot(orders_ct.index, orders_ct.values, alpha=0.8, color=color[1])
plt.ylabel('Número de ocorrências', fontsize=12)
plt.xlabel('Eval_set', fontsize=12)
plt.title('Número de registros em cada dataset', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

# Por produtos
orders_prior_train_df.user_id.unique().shape[0]
orders_prior_train_df.order_id.unique().shape[0]
orders_prior_train_df.product_id.unique().shape[0]

orders_prior_train_df.describe(include=[np.number])

orders_ct = orders_prior_train_df[['order_id', 'product_id']].groupby('order_id', as_index=False).product_id.count()
orders_ct.head()
orders_ct[['product_id']].describe()

orders_resume = orders_prior_train_df.groupby(['user_id', 'product_id']).size().reset_index(name='rating')
orders_resume_shapeInicial = orders_resume.shape
orders_resume.shape
orders_resume.head(5)
orders_resume.columns
print('users %s' % orders_resume.user_id.unique().shape[0])
print('items %s' % orders_resume.product_id.unique().shape[0])


# Usuário 1 - distribuição dos produtos
orders_ct = orders_resume[orders_resume['user_id'].isin([1])]
orders_ct = orders_ct.sort_values('rating', ascending=False)
plt.figure(figsize=(12,8))
sns.barplot(orders_ct.product_id, orders_ct.rating, alpha=0.8, color=color[1])
plt.ylabel('rating', fontsize=12)
plt.xlabel('product_id', fontsize=12)
plt.title('Quantidade de produtos comprados pelo user_id=1', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

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


orders_resume.head()
userItemRatingMatrix=pd.pivot_table(orders_resume, values='rating', index=['user_id'], columns=['product_id'])
userItemRatingMatrix.head(10)
userItemRatingMatrix.iloc[1:20,1:20].head()
userItemRatingMatrix.shape

# Mostra a listagem de produtos do K-means para  usuário 90 com seus corredores
a_products = [28746, 20120, 39476, 6219, 19895, 10769, 36957, 118, 25838, 14771]
products_df[['product_id', 'product_name', 'aisle']][products_df['product_id'].isin(a_products)]

# Mostra a listagem de produtos do k-nearest neighbors para  usuário 90 com seus corredores
a_products = [21387, 49684, 1464, 10961, 45438, 26318, 5777, 19141, 47157, 18812]
products_df[['product_id', 'product_name', 'aisle']][products_df['product_id'].isin(a_products)]



# Limpa memória
gc.collect()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 23:08:30 2017

@author: amoghjahagirdar
"""

from sklearn.decomposition import PCA
import pandas as pd



min_variance_explained = 0.95

pca = PCA(n_components = min_variance_explained)


if __name__ == "__main__":
    #train_data = pd.read_csv('train.csv')
    #test_data = pd.read_csv('test.csv')
    #train_data = train_data.fillna(method = 'ffill')
    #transformed_train_data = pca.fit_transform(train_data)

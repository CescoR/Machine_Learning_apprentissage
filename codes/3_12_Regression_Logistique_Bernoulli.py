# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:23:52 2023

@author: romeofr
"""

# test de la fonction de vraisemblance de Bernoulli
# fonction de vraisemblance pour la distribution de Bernoulli
def likelihood(y, yhat):
    return yhat * y + (1 - yhat) * (1 - y)

# test pour y=1
y, yhat = 1, 0.9
print('y=%.1f, yhat=%.1f, likelihood: %.3f' % (y, yhat, likelihood(y, yhat)))

y, yhat = 1, 0.1
print('y=%.1f, yhat=%.1f, likelihood: %.3f' % (y, yhat, likelihood(y, yhat)))

# test for y=0
y, yhat = 0, 0.1
print('y=%.1f, yhat=%.1f, likelihood: %.3f' % (y, yhat, likelihood(y, yhat)))

y, yhat = 0, 0.9
print('y=%.1f, yhat=%.1f, likelihood: %.3f' % (y, yhat, likelihood(y, yhat)))
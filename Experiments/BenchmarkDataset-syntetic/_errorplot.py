# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 00:40:40 2020

@author: Henock
"""


import numpy as np
import matplotlib.pyplot as plt

# # example data
# x = np.arange(0, 7, 1)
# y = [58.0,16.3,92.7,124.3,176.0,181.9,219.6]

# # example variable error bar values
# yerr = [0.01,0.00,0.15,0.00,1.68,0.00,0.00]
# # xerr = 0.1 + yerr

# plt.errorbar(x, y, yerr=yerr, fmt='o')



# aggregation dataset
x = ['K-means',	'FCM',	'P-POCS', 	'APOCS']
y = [92.70310713,	94.30223982,	93.49967813,	93.76035394]
yerr= [0.151684345,	1.927277908,	2.46E-09,	0.322636065]
plt.ylabel('Error')
plt.xlabel('Algorithm')
plt.title('Aggregation dataset')
plt.errorbar(x, y, yerr, fmt='.k')
plt.savefig('_aggregation.jpg')
plt.show()


# a2☻
x = ['K-means',	'FCM',	'P-POCS', 	'APOCS']
y = [176.0403671,	225.218981,	181.8820362,	190.2157082]
yerr= [1.682281294,	15.5207021,	2.26E-08,	0.164445496]
plt.ylabel('Error')
plt.xlabel('Algorithm')
plt.title('a2 dataset')
plt.errorbar(x, y, yerr, fmt='.k')
plt.savefig('_a2.jpg')
plt.show()


# flame
x = ['K-means',	'FCM',	'P-POCS', 	'APOCS']
y = [58.03528573,56.83141515	,58.87564779,58.93228417]
yerr= [0.014120366	,1.44E-07	,1.22E-06	,0.278328683]
plt.ylabel('Error')
plt.xlabel('Algorithm')
plt.title('Flame dataset')
plt.errorbar(x, y, yerr, fmt='.k')
plt.savefig('_flame.jpg')
plt.show() 


# R15
x = ['K-means',	'FCM',	'P-POCS', 	'APOCS']
y = [16.34140008,19.59476412	,16.42273389,16.47371361]
yerr= [7.37E-15	,6.993348274	,1.69E-12	,0.040342458]
plt.ylabel('Error')
plt.xlabel('Algorithm')
plt.title('R15 dataset')
plt.errorbar(x, y, yerr, fmt='.k')
plt.savefig('_R15.jpg')
plt.show()


# a1
x = ['K-means',	'FCM',	'P-POCS', 	'APOCS']
y = [124.3228771,135.4462462	,130.6988145,124.8178616]
yerr= [0.00794762	,5.063520466	,1.55E-05	,0.062347744]
plt.ylabel('Error')
plt.xlabel('Algorithm')
plt.title('a1 dataset')
plt.errorbar(x, y, yerr, fmt='.k')
plt.savefig('_a1.jpg')
plt.show() 

# s1
x = ['K-means',	'FCM',	'P-POCS', 	'APOCS']
y = [181.9665741,212.298576	,182.9002396,183.1624869]
yerr= [0.002510439	,29.08118756	,9.90E-09	,0.216309459]
plt.ylabel('Error')
plt.xlabel('Algorithm')
plt.title('a1 dataset')
plt.errorbar(x, y, yerr, fmt='.k')
plt.savefig('_s1.jpg')
plt.show() 


# s2
x = ['K-means',	'FCM',	'P-POCS', 	'APOCS']
y = [219.6049336,228.3850464	,220.9238978,221.1709417]
yerr= [0.002276605	,13.61732418	,8.57E-07	,0.225142719]
plt.ylabel('Error')
plt.xlabel('Algorithm')
plt.title('a1 dataset')
plt.errorbar(x, y, yerr, fmt='.k')
plt.savefig('_s2.jpg')
plt.show()



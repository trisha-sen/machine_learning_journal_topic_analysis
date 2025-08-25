# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:36:22 2020

@author: Trisha
"""
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from time import time
import pandas as pd
import seaborn as sns
MAX_YEAR = 2025
MIN_YEAR = 2021
ni, nn = 1,3

#%% Plot yearly trends
def YearlyTrendsPlot(dSup, P):
    avgP = np.mean(P, axis=1) #np.ones([(2020-1996),1])*
    
    Years = range(MIN_YEAR,MAX_YEAR+1)

    mm = P.shape[1]//(nn*ni) + min(1, P.shape[1]%(nn*ni))
    fig, axes = plt.subplots(ni,nn,figsize=(5*nn,5*ni),tight_layout = True)
    ax = axes.flatten()
    
    start = 0
    for s in range(nn*ni):
        end = min(start + mm, P.shape[1])
        for i in dSup.index[start: end]:
            # if i==dSup.index[0]:
            #     continue
#             if i==1 or i==3 or i==7 or i==9 or i==6 or i==0:
            ax[s].plot(Years, P[:,i], '-', label=str(i)+' '+str(dSup.topWords[i][:5]), lw = 1.5)
#             else:
#                 ax[s].plot(Years,P[:,i],'-',label=str(i)+' '+str(dSup.topWords[i][:1]),alpha = 0.3, lw = 2)
        ax[s].plot(Years,avgP,'k--')
        ax[s].set_xlabel('Year', fontsize=15)
        ax[s].set_ylabel('Topic Popularity', fontsize=15)
        ax[s].set_ylim([0, 0.14])
        ax[s].legend(loc='lower left', 
                     bbox_to_anchor= (0.0, 1.01),
                     ncol=1, borderaxespad=0,
                     frameon=False, fontsize=8
                    )
        start = end
    plt.show()
    return

#%% WOrdcloud for overall topics
def TopicWordClout(dSup):
    # Create a list of word
    text = []
    freq = []
    for i in range(len(dSup)):
        for j in range(3):
            text.append(dSup.loc[i].topWords[j])
            freq.append(float(dSup.loc[i].Popularity))
            
    freq = np.hstack(freq)
    text = np.hstack(text)
    dictionary = dict(zip(text, freq))
    
    # Create the wordcloud object
    wordcloud = WordCloud(width=480*3, height=480*2, margin=0, colormap='autumn',max_font_size=150, min_font_size=30,
                          background_color='black').fit_words(dictionary)
    
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=10, y=10)
    plt.show()
    
#%% Plot the number of authors per paper
def CollaborationPlot(dfPlot):    
    dfPlot['#Authors'] = dfPlot['Authors'].str.split(";").str.len().astype(float)
    plt.figure(figsize=(5,3))
    ax = sns.boxplot(x="Year", y="#Authors",
                  data=dfPlot, showfliers=False)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Authors / Paper', fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    plt.show()
    
def citation_plot(
    dfPlot, dSup, avgTot,
    argsAll,
    valsAll,
    n_components
):
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    from matplotlib import cm
    import math

    YEAR2 = range(MIN_YEAR,MAX_YEAR+1)
    # ni, nn = 2,2
    mm = n_components//(nn*ni) + min(1, n_components%(nn*ni))
    fig2, axes2 = plt.subplots(ni,nn,figsize=(5*nn,5*ni),tight_layout = True)
    axs2 = axes2.flatten()

    tol = 0.33
    marker = ['.','x','^','3','_']

    Topics = list(dSup.index) #np.arange(0,n_components) #[0,1,2,3,4,6,7,8,9,11]
    mydata = []
    # nCm = 0;
    counter = 0
    for topic in Topics:  
        # nCm = counter - math.floor(counter/mm)*mm
        # color=cm.tab10(nCm)
        ind = []
        for i in range(1):
            topicInds  = (np.argwhere(argsAll[:,i] == topic))
            if topicInds != []:
                topicInds = np.concatenate(topicInds)
            feasInds = (np.argwhere(valsAll[:,i] >= tol))
            if feasInds != []:
                feasInds = np.concatenate(feasInds)
            ind.append((np.intersect1d(topicInds, feasInds)))

    #     print(str(dSup.topWords[topic][:3]))
        ind = np.hstack(ind)

        dfTemp = dfPlot.loc[ind]
        UniqueAuthors = dfTemp['Authors'].str.split(";").str[-1].unique()
        totAuth = len(UniqueAuthors)

        UB = np.arange(0.05,1,0.05)
        nAuth = []
        for upperBound in UB:
            lowerBound = upperBound - 0.05
            df0New = dfTemp['Annual Citations'].sort_values(ascending=False).cumsum()
            df0New = df0New[(df0New <= upperBound*np.sum(dfTemp['Annual Citations']))]
            df0New = dfTemp.loc[df0New.index]
            CitedUniqueAuthors = df0New['Corresponding Author'].unique()
            nAuth.append(len(CitedUniqueAuthors))

        ii = math.floor(counter//mm)
    #     axs[ii].plot(np.hstack(nAuth)/totAuth*100,(UB-0.025)*100,'-o',label=str(topic)+' '+str(dSup.topWords[topic][:1]),
    #                 color=color,marker=marker[nCm])

#         mydata.append(dfTemp[dfTemp['Annual Citations'] <=30]['Annual Citations'])
        avg = []
        year2 = []
        for year in YEAR2:
            df0 = dfTemp[(dfTemp.Year == year)]
            if len(df0) > 0:
                year2.append(year)
                Citations = df0['Cited by']
                avg.append(round(np.mean(Citations)/(MAX_YEAR+1-year),2))
        spl = UnivariateSpline(year2, avg, k=min(len(avg)-1, 4))
        spl.set_smoothing_factor(1)
        axs2[ii].plot(year2, spl(year2), '-o', lw = 1, markersize=3,
                        label=str(topic)+' '+str(dSup.topWords[topic][:6]))
        counter+=1

    for ii in range(nn*ni):
        axs2[ii].plot(YEAR2, avgTot,'k--')
        axs2[ii].tick_params(axis="x", labelsize=10)
        axs2[ii].tick_params(axis="y", labelsize=10)
        axs2[ii].set_xlabel('Year', fontsize=12)
        axs2[ii].set_ylabel('Impact', fontsize=12)
        axs2[ii].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=1, 
                borderaxespad=0, frameon=False, fontsize=7)
        # axs2[ii].grid(which='major', axis='both')

    plt.show()
    return


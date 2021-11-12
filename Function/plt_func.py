#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import umap.umap_ as umap
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

def plt_umap(Feature_output, y, save_direct)
    mapper = umap.UMAP(random_state=42, n_neighbors = 10, n_components = 2).fit(Feature_output[0])
    embedding0 = mapper.transform(Feature_output[0])
    embedding1 = mapper.transform(Feature_output[1])
    embedding2 = mapper.transform(Feature_output[2])

    ########## ラベル定義  #############
    Emotion_labels = ["Negative", "Neutral", "Positive"]
    Domain_labels = ["Source", "Target", "Test"]
    
    ########## embedとｙを格納 ##########
    embed_dict = {"Source": embedding0, "Target": embedding1, "Test": embedding2}
    y_dict = {"Source": y[0], "Target": y[1], "Test": y[2]}

    ########## ラベル関連定数  ##########
    __colors_dict = {"Source": "red", "Target": "blue", "Test": "green"}
    __shapes_dict = {"Negative": "v", "Neutral": "^", "Positive": "o"}
    __label_dict = {"Negative": 0, "Neutral": 1, "Positive": 2}
    Emotion_handles = []
    Domain_handles = []

    #############  サイズ定義  #######################
    size = 120
    plt.figure(figsize=(7, 7))
    #############  感情の凡例づくり  ###############
    handles1=[]
    for class_name, shapes in __shapes_dict.items():
        space_0 = embed_dict["Target"]
        space_1 = space_0[y_dict["Target"] == __label_dict[class_name]]
        p=plt.scatter(x=space_1[:,0], y=space_1[:,1], alpha=0.9, 
                    c="black", marker=shapes)
        handles1.append(p)
        p.remove()

    #############  ドメインの凡例づくり  #############
    handles2=[]
    for domain_name, embed in embed_dict.items():
        space_2 = embed[y_dict[domain_name]==1] #感情ラベルはなんでもよい
        p=plt.scatter(x=space_2[:,0], y=space_2[:,1], alpha=0.9, 
                    c=__colors_dict[domain_name], marker=",")
        handles2.append(p)
        p.remove()

    #############  実際のプロット  ################
    for domain_name, embed in embed_dict.items():
        for class_name, shapes in __shapes_dict.items():
            emospace = embed[y_dict[domain_name] == __label_dict[class_name]]
            plt.scatter(x=emospace[:, 0], y=emospace[:,1], alpha=0.9, c=__colors_dict[domain_name],
                        label=shapes, s=size, linewidths=1, edgecolors="black")

    plt.xlabel("Embedding Dimension 1", fontsize=18), plt.ylabel("Embedding Dimention 2", fontsize=18), plt.tick_params(labelsize=18)
    legend = plt.legend(handles1, Emotion_labels, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=20, handletextpad=0.01)
    legend2 = plt.legend(handles2, Domain_labels, bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0, fontsize=20, handletextpad=0.01)
    plt.gca().add_artist(legend)
    plt.savefig(save_direct+'Proposed_'+str(exp+1)+'_'+str(count)+'.pdf', bbox_extra_artists=(legend,legend2), bbox_inches="tight", pad_inches=0)
    plt.close()
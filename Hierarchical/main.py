import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch



def draw_hist(list,cl_amt):

    list.sort(key=lambda i: i[0])

    # print(list)
    list_occ = []
    last_artist = list[0][0]
    j = 0

    list_occ.append([list[0], 1])
    l = len(list)
    rng = range(1, l)

    for i in rng:
        artist = list[i][0]
        if artist == last_artist:
            list_occ[j][1] = list_occ[j][1] + 1
            list_occ[j][0].append(list[i][1])
        else:
            j = j + 1
            list_occ.append([list[i], 1])
        last_artist = artist
    list_occ.sort(key=lambda i: i[1], reverse=True)
    # print(list_occ)

    num_of_clusters = 4
    l = len(list_occ)
    for i in range(0, l):
        cl_occ = [0]*cl_amt
        l2 = len(list_occ[i][0])
        for j in range(1, l2):
            cl_occ[list_occ[i][0][j]] = cl_occ[list_occ[i][0][j]] + 1
        m = max(cl_occ)
        artist = list_occ[i][0][0]
        list_occ[i][0] = [artist, m / (l2 - 1)]
    top=20
    list_occ=list_occ[0:top]

    ch_list=[0]*top
    for i in range(0,top):
        ch_list[i]=list_occ[i][0][1]

    plt.hist(ch_list)
    plt.show()
    #print(ch_list)



def my_dendrogram(X_den,labels):





    links =sch.linkage(X_den,method='ward')
    d=sch.dendrogram(links, labels=labels,
                    leaf_rotation=90.,
                    leaf_font_size=8.,)
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv("data.csv")
    #print(df.head(1))
    chosen = ["energy", "liveness", "tempo", "valence", "loudness", "speechiness", "acousticness", "danceability",
              "instrumentalness"]

    X = df[chosen].values
    song_title = df["song_title"].values
    song_title.shape = (2017, 1)
    artist = df["artist"].values
    artist.shape = (2017, 1)


    num_of_songs = 2017
    X = X[0:num_of_songs, :]
    song_title=song_title[0:num_of_songs,:]
    artist=artist[0:num_of_songs,:]




    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)



    labels=[""]*len(artist)
    for i in range(0,len(artist)):
        labels[i] = artist[i] + " - " + song_title[i]
    labels=labels[500:540]
    X_den = X[500:540]
    #my_dendrogram(X_den,labels)



    clusters_amt=15
    hc = AgglomerativeClustering(n_clusters=clusters_amt)



    predict=hc.fit_predict(X)

    predict.shape=(num_of_songs,1)

    pca = PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)



    X_c=np.append(X, artist, axis=1)
    X_c = np.append(X_c, predict, axis=1)
    sorted_array = X_c[:,3:5][np.argsort(X_c[:,3:5][:, 1])]
    sorted_list=sorted_array.tolist()
    #draw_hist(sorted_list,clusters_amt)

    out = pd.DataFrame(sorted_list)
    out.to_csv('Hc15.csv')


    X = np.append(X, predict, axis=1)

    X_d=pd.DataFrame(X)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X_i=clusters_amt*[None]
    for i in range(0, clusters_amt):
        X_i[i]=X[X_d[3]==i]


    for i in range(0,clusters_amt):
        x=X_i[i][:,0]
        y = X_i[i][:, 1]
        z = X_i[i][:, 2]
        ax.scatter(x, y,z)

    fig.show()

if __name__ == '__main__':
    main()



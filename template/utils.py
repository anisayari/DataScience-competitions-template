import pandas as pd
from tqdm import tqdm
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import configparser

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import time
timestr = time.strftime("%Y%m%d-%H%M%S")


#pd.options.display.float_format = '{:.5g}'.format
pd.options.display.float_format = '{:,.2f}'.format

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib, pickle


def normalize_by(df, by):
    by_list = df[by].unique()
    for by_ in tqdm(by_list):
        ids = df[df[by] == by_].index
        scaler = MinMaxScaler()
        df.loc[ids, features_n] = scaler.fit_transform(df.loc[ids, features_n])
    return df

def fillInf(df, val):
    numcols = df.select_dtypes(include='number').columns
    cols = numcols[numcols != 'winPlacePerc']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    for c in cols: df[c].fillna(val, inplace=True)
    return df

def reduce_mem_usage(props,prompt=True):
    nan_cols=props.columns[props.isnull().any()].tolist()
    if prompt:
        start_mem_usg = props.memory_usage().sum() / 1024**2 
        print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            #if prompt:
                # Print current column type
                #print("******************************")
                #print("Column: ",col)
                #print("dtype before: ",props[col].dtype)
                #pass
            if col in nan_cols:
                if prompt: 
                    print('Column: %s has NAN values'%col)
                props.loc[:,col] = props.loc[:,col].astype(np.float32)
            else:
                # make variables for Int, max and min
                IsInt = False
                mx = props[col].max()
                mn = props[col].min()
                
                # Integer does not support NA, therefore, NA needs to be filled
                

                # test if column can be converted to an integer
                asint = props[col].astype(np.int64)
                result = (props[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True

                
                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if mn >= 0:
                        if mx < 2**8:
                            props.loc[:,col] = props.loc[:,col].astype(np.uint8)
                        elif mx < 2**16:
                            props.loc[:,col] = props.loc[:,col].astype(np.uint16)
                        elif mx < 2**32:
                            props.loc[:,col] = props.loc[:,col].astype(np.uint32)
                        else:
                            props.loc[:,col] = props.loc[:,col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            props.loc[:,col] = props.loc[:,col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            props.loc[:,col] = props.loc[:,col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            props.loc[:,col] = props.loc[:,col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            props.loc[:,col] = props.loc[:,col].astype(np.int64)    
                
                # Make float datatypes 32 bit
                else:
                    props.loc[:,col] = props.loc[:,col].astype(np.float32)

            #if prompt:
                # Print new column type
                #print("dtype after: ",props[col].dtype)
                #print("******************************")
                #pass
    
    if prompt:
        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = props.memory_usage().sum() / 1024**2 
        print("Memory usage is: ",mem_usg," MB")
        print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props



def get_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config


# DATA PREPARATION
def compile_data(file, max_index):
    list_ = []
    print('[INFO] Start {}...'.format(file))
    for i in tqdm(range(1, max_index + 1)):
        if i < 10:
            df = pd.read_csv('../data/{}_0{}.csv'.format(file, i), sep=";", header=0, low_memory=False,
                             encoding="ISO-8859-1")
        else:
            df = pd.read_csv('../data/{}_{}.csv'.format(file, i), sep=";", header=0, low_memory=False,
                             encoding="ISO-8859-1")
        list_.append(df)

    print('[INFO] Concatenate in progress..')
    frame = pd.concat(list_, axis=0, ignore_index=True)
    print('[INFO] Concatenate done')

    print('[INFO] Saving in progress..')
    frame.to_csv('../data/{}.csv'.format(file), header=True, sep=";", index=False)
    print('[INFO] Saving done')


def extract_data(df_path, year_to_select):
    print('[INFO] Reading csv {}...'.format(df_path))
    df = pd.read_csv(df_path, sep=";", header=0, low_memory=False)
    print('[INFO] Reading done, shape {}'.format(df.shape))

    print('[INFO] Selecting sub df...')
    file_name = os.path.splitext(os.path.basename(df_path))[0]
    if file_name == "EDVIE_INM_CAPIDEF":
        sub_df = df[df['PASTRF'] == year_to_select - 1]
    else:
        sub_df = df[df['PASTRF'] == year_to_select]

    print('[INFO] Selecting done, shape {}'.format(sub_df.shape))

    print('[INFO] Saving...')
    sub_df.to_csv(os.path.dirname(df_path) + '/{0}/{1}.csv'.format(year_to_select, file_name), header=True, sep=";",
                  index=False)
    print('[INFO] Saving done')


# DATA EXPLORATION

def list_int_and_float_columns(df):
    list_col_to_plot = []
    for col_ in df.columns:
        if df[col_].dtype == int or df[col_].dtype == float:
            list_col_to_plot.append(col_)
    return list_col_to_plot


def print_value_counts_of_all_columns(df, bins=None, normalize=True):
    list_col_to_plot = list_int_and_float_columns(df)
    # print value counts
    for col_ in df.columns:
        print('{}'.format(col_))
        if col_ in list_col_to_plot:
            if bins is not None:
                print(df[col_].value_counts(normalize=normalize, dropna=False, bins=bins))
            else:
                print(df[col_].value_counts(normalize=normalize, dropna=False))

        else:
            print(df[col_].value_counts(normalize=normalize, dropna=False))
        print('---------------------')


def check_and_sort_long_to_plot(df, list_col_to_plot, max, sample, sample_size):
    for col_ in tqdm(list_col_to_plot):
        shape_ = df[col_].value_counts().reset_index().shape[0]
        if shape_ > max and sample:
            print('[INFO] May be really long to plot for {} (SHAPE:{})'.format(col_, shape_))
            if sample:
                print('[INFO] May be really long to plot for {} (SHAPE:{})'.format(col_, shape_))
                list_col_to_plot[list_col_to_plot.index(col_)] = (col_, sample_size)
            else:
                list_col_to_plot[list_col_to_plot.index(col_)] = (col_, None)
                # print('[INFO] Removing {} from list to plot'.format(col_))
                # list_col_to_plot.remove(col_)
        else:
            list_col_to_plot[list_col_to_plot.index(col_)] = (col_, None)

    return list_col_to_plot


def print_multiple(df, kind, sample=False, sample_size=None, max=1000, save_fig=False, save_fig_path=None):
    """
    print multiple  plot (distplot or countplot) using conditional in int and float columns
    All is plotted in one figures.
    :param df: Dataframe
    :return: figure
    """
    if save_fig and save_fig_path == None:
        raise ValueError('Please define the path of the fig')

    if sample and sample_size == None:
        raise ValueError('Please define the size of the sample')

    print('[INFO] Print in progress')
    if kind == 'distplot' or kind == 'boxplot' or kind == "residplot":
        list_col_to_plot = list_int_and_float_columns(df)
    else:
        list_col_to_plot = df.columns.tolist()

    list_col_to_plot = check_and_sort_long_to_plot(df, list_col_to_plot, max=max, sample=sample,
                                                   sample_size=sample_size)

    # print distplot
    sns.set(style='darkgrid')
    if len(list_col_to_plot) == 0:
        print('[INFO - {}] No numerical columns to plot'.format(len(list_col_to_plot)))
    if len(list_col_to_plot) > 1:
        fig, list_ax = plt.subplots(math.ceil(len(list_col_to_plot) / 2), 2, figsize=(30, 5 * len(list_col_to_plot)))
        list_ax_tmp = []
        for (ax1, ax2) in list_ax:
            list_ax_tmp.append(ax1)
            list_ax_tmp.append(ax2)
        list_ax = list_ax_tmp
    elif len(list_col_to_plot) == 1:
        fig, list_ax = plt.subplots(1, 1)
        print(list_ax)
        list_ax = [list_ax]
        print(list_ax)
    else:
        print('[INFO] This case is unknow')

        list_ax = []

    if kind == 'residplot':
        count = 0
        for col2, in list_col_to_plot:
            for col1 in list_col_to_plot:
                if col1 == col2:
                    continue
                y = df[col2]
                x = df[col1]
                sns.residplot(x, y, lowess=True, color="g", ax=list_ax[count])
                sns.lmplot(x=col1, y=col2, data=df, ax=list_ax[count])
                count += 1
    else:
        for i, (col_, sample_size_) in tqdm(enumerate(list_col_to_plot)):
            if sample_size_ is not None:
                to_plot = df[col_].dropna().sample(sample_size_)
            else:
                to_plot = df[col_].dropna()
            if kind == 'distplot':
                sns.distplot(to_plot, kde=False, ax=list_ax[i], bins='auto', rug=True)
            elif kind == 'boxplot':
                sns.boxplot(x=to_plot, ax=list_ax[i])
            elif kind == 'countplot':
                sns.countplot(to_plot, ax=list_ax[i])

    # fig.savefig('test.png', bbox_inches="tight")


def encoded_columns(df, columns_to_encode_list, name_df):
    le = preprocessing.LabelEncoder()

    from collections import defaultdict
    d = defaultdict(preprocessing.LabelEncoder)
    fit = df[columns_to_encode_list].apply(lambda x: d[x.name].fit_transform(x.fillna('0')))
    for col in fit.columns:
        df[col + '_LABELED'] = fit[col]
        """
        outF = open("label_map\{0}_{1}_label_map.csv".format(name_df, col), "w")
        outF.write('label_encoded;label_decoded')
        for i, class_ in enumerate(d[col].classes_):
            outF.write('{};{}'.format(i, class_))
            outF.write("\n")
        outF.close()
    for key, value in d.items():
        print(key, value)
        """
    return df


def PCA_study(df):
    # number of variable
    p = df.shape[1]
    print('[INFO] Number of Variable: {}'.format(p))
    # nombre d'observations
    n = df.shape[0]
    print('[INFO] Number of Variable: {}'.format(n))

    # print(data.columns)
    print('[INFO] DATA DONE')
    # data.info()

    print('[INFO] Scaling in progress...')
    min_max_scaler = preprocessing.StandardScaler()
    # min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    print('[INFO] Check if means stay near 0')
    print(np.mean(np_scaled, axis=0))
    print('[INFO] Check if standars deviation is equal at 1')
    print(np.std(np_scaled, axis=0, ddof=0))

    data = pd.DataFrame(np_scaled)
    print('[INFO] SCALING DONE')
    sns.set(style='darkgrid')

    print('[INFO] PCA in progress...')

    pca = PCA(svd_solver='full')
    data = pca.fit_transform(data)
    index_columns = ['PCA-{}'.format(i) for i in range(1, p + 1)]
    pca_components_df = pd.DataFrame(pca.components_, columns=df.columns, index=index_columns)
    print('[INFO] PCA DONE')

    # standardize these 3 new features
    print('[INFO] Standardization in progress...')
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)
    print('[INFO] Standardization DONE')

    print('[INFO] Explained Variance')
    eigval = (n - 1) / n * pca.explained_variance_
    print(eigval)
    # proportion de variance expliquee
    print(pca.explained_variance_ratio_)
    # scree plot
    plt.plot(np.arange(1, p + 1), eigval)
    plt.title("Scree plot")
    plt.ylabel("Eigen values")
    plt.xlabel("Factor number")
    plt.savefig('data/output/{}_Scree_plot.png'.format(timestr))
    plt.show()
    plt.close()
    # cumul de variance expliquee
    plt.plot(np.arange(1, p + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.title("Explained variance vs. # of factors")
    plt.ylabel("Cumsum explained variance ratio")
    plt.xlabel("Factor number")
    plt.savefig('data/output/{}_Explained_variance.png'.format(timestr))
    plt.show()
    plt.close()
    # seuils pour test des batons brises
    bs = 1 / np.arange(p, 0, -1)
    bs = np.cumsum(bs)
    bs = bs[::-1]
    # test des batons brises
    print(pd.DataFrame({'Val.Propre': eigval, 'Seuils': bs}))

    # contribution des individus dans l'inertie totale
    di = np.sum(data ** 2, axis=1)
    print(pd.DataFrame({'ID': df.index, 'd_i': di}).sort_values(['d_i'], ascending=False).head(10))

    # qualite de representation des individus - COS2
    cos2 = np_scaled ** 2
    for j in range(p):
        cos2[:, j] = cos2[:, j] / di
    print(pd.DataFrame({'id': df.index, 'COS2_1': cos2[:, 0], 'COS2_2': cos2[:, 1]}).head())

    # verifions la theorie - somme en ligne des cos2 = 1
    print(np.sum(cos2, axis=1))

    # contributions aux axes
    ctr = np_scaled ** 2
    for j in range(p):
        ctr[:, j] = ctr[:, j] / (n * eigval[j])

    print(pd.DataFrame({'id': df.index, 'CTR_1': ctr[:, 0], 'CTR_2': ctr[:, 1]}).head())

    # representation des variables
    # racine carree des valeurs propres
    sqrt_eigval = np.sqrt(eigval)

    # correlation des variables avec les axes
    corvar = np.zeros((p, p))
    for k in range(p):
        corvar[:, k] = pca.components_[k, :] * sqrt_eigval[k]

    # afficher la matrice des correlations variables x facteurs
    print('[INFO] Correlation matrices (variable x factor)')
    print(corvar)

    # on affiche pour les deux premiers axes
    print(('[INFO] plot first two axes'))
    print(pd.DataFrame({'id': df.columns, 'COR_1': corvar[:, 0], 'COR_2': corvar[:, 1]}).head())

    # cercle des correlations
    fig, axes = plt.subplots(figsize=(15, 15))
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)
    # affichage des etiquettes (noms des variables)
    for j in range(p):
        plt.annotate(df.columns[j], (corvar[j, 0], corvar[j, 1]))

    # ajouter les axes
    plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
    plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

    # ajouter un cercle
    cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
    axes.add_artist(cercle)
    # affichage
    fig.savefig('data/output/{}_PCA.png'.format(timestr))
    plt.show()
    plt.close()

    for index_ in pca_components_df.index:
        print('------------- {} -------------'.format(index_))
        dict_ = {}
        for col_ in pca_components_df.columns:
            dict_[col_] = pca_components_df.loc[index_, col_]
        d_view = [(v, k) for k, v in dict_.items()]
        d_view.sort(reverse=True)  # natively sort tuples by first element
        for v, k in d_view:
            pass
            # print("{}: {}".format(k,v))
        plt.figure(figsize=(15, 6))
        sns.barplot(pd.Series(list(zip(*d_view))[1]), pd.Series(list(zip(*d_view))[0]), palette="BuGn_r").set_title('PCA-{0}_importance_variable'.format(index_))
        plt.xticks(rotation=70)
        plt.savefig('data/output/{0}_PCA-{1}_importance_variable.png'.format(timestr, index_))
        plt.show()

    return data, pca_components_df

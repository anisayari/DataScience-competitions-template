import urllib

import cv2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import configparser
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import scale, MinMaxScaler, LabelEncoder
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
#pd.options.display.float_format = '{:.5g}'.format
pd.options.display.float_format = '{:,.2f}'.format

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib, pickle

from stop_words import get_stop_words
stop_words_fr = get_stop_words('fr')


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


def encoded_columns(df, columns_to_encode_list):
    le = preprocessing.LabelEncoder()

    from collections import defaultdict
    d = defaultdict(preprocessing.LabelEncoder)
    fit = df[columns_to_encode_list].apply(lambda x: d[x.name].fit_transform(x.fillna('missing')))
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


def extract_features_from_image(df, id_column, column_path):
    print('here______________________________')
    from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
    import tensorflow.keras.backend as K
    import tensorflow as tf
    from tensorflow.python.client import device_lib

    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.Session()

    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    print(get_available_gpus())

    def load_image(img_size, path='', url=''):
        def resize_to_square(im, img_size):
            old_size = im.shape[:2]  # old_size is in (height, width) format
            ratio = float(img_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            # new_size should be in (width, height) format
            im = cv2.resize(im, (new_size[1], new_size[0]))
            delta_w = img_size - new_size[1]
            delta_h = img_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            color = [0, 0, 0]
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            return new_im
        if url =='':
            image = cv2.imread(path)
            
        elif path=='': 
                # download the image, convert it to a NumPy array, and then read
            # it into OpenCV format
            resp = urllib.request.urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        new_image = resize_to_square(image, img_size)
        new_image = preprocess_input(new_image)
        return new_image

    def init_densenet():
        print('[INFO] Init Densenet...')
        inp = Input((256, 256, 3))
        print('[INFO] import Densenet...')
        backbone = DenseNet121(input_tensor=inp, include_top=False,
                               weights='../input/densenet-121-weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
        print('[INFO] import Densenet DONE')
        x = backbone.output
        x = GlobalAveragePooling2D()(x)
        x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
        x = AveragePooling1D(4)(x)
        out = Lambda(lambda x: x[:, :, 0])(x)
        m = Model(inp, out)
        print('[INFO] Init Densenet DONE.')
        return m

    m = init_densenet()

    print('[INFO] Start Image Features_Extraction...')
    img_size = 256
    batch_size = 16
    ids = df[id_column].values
    n_batches = len(ids) // batch_size + 1
    features = {}
    for b in tqdm(range(n_batches)):
        start = b * batch_size
        end = (b + 1) * batch_size
        batch_ids = ids[start:end]
        batch_images = np.zeros((len(batch_ids), img_size, img_size, 3))
        for i, id_ in enumerate(batch_ids):
            
            #image_name = '{}-{}.jpg'.format(id_, 1)
            #image_path = jp(input_dir, subfolder, image_name)
            image_path= df.loc[df['id']==id_][column_path].values[0]
            try:
                batch_images[i] = load_image(256,url=image_path)
            except:
                continue
        batch_preds = m.predict(batch_images)
        for i, id_ in enumerate(batch_ids):
            features[id_] = batch_preds[i]

    df_features = pd.DataFrame.from_dict(features, orient='index')
    df_features.rename(columns=lambda k: 'img_{}'.format(k), inplace=True)
    df_features.reset_index(inplace=True)
    df_features.rename(columns={df_features.columns[0]: id_column}, inplace=True)
    n_components = 200
    svd = TruncatedSVD(n_components=n_components)
    X = df_features[['img_{}'.format(k) for k in range(256)]].values
    svd.fit(X)
    print('fit done')
    X_svd = svd.transform(X)
    X_svd = pd.DataFrame(X_svd, columns=['img_svd_{}'.format(i) for i in range(n_components)])
    X_svd[id_column] = df.id.values.tolist()

    df = pd.concat([df.set_index(id_column), X_svd.set_index(id_column)], sort=False, axis=1).reset_index()
    df.rename(columns={df.columns[0]: id_column}, inplace=True)
    print('[INFO] Image Features_Extraction DONE.')
    return df



"""
IMAGE
"""
#@TODO : 'To fill'

"""
SONG
"""
#@TODO : 'To fill'


"""
FEATURE ENGINEERING COMMON FUNCTIONS
"""
#@TODO : 'Need to check with auto FE libraries
"""
MATHEMATICS FEATURES
"""
def create_mathematics_features(df, column_to_count, column_to_groupby):
    df_tmp = df.groupby(column_to_groupby)[column_to_count].agg(['count','mean', 'std', 'max', 'min'])
    df_tmp.columns =['count_' + column_to_count, 'mean_' + column_to_count, 'std_' + column_to_count,'max_' + column_to_count, 'min_' +column_to_count,]
    df = df.merge(df_tmp, on=column_to_groupby, how='left')
    return df 

"""
NUMERICAL FEATURES
"""
def get_len_columns(df, len_columns):
    for col_ in len_columns:
        df["len_" + col_] = df[col_].str.len()
    return df

def transform_to_log(df,columns_to_log):
    for col_ in columns_to_log:
        df['log_' + col_] = (1+df[col_]).apply(np.log)
    return df

def count_product_per_store(df, column_to_groupby, column_to_count):
    tmp = df.groupby(column_to_groupby).count()[column_to_count].reset_index()
    tmp.columns = [column_to_groupby] + ["number_" + column_to_count + '_' + column_to_groupby]
    df = df.merge(tmp, on=column_to_groupby, how='left')
    return df

def count_item_column(df, column_to_count, column_groupby):
    rescuer_count = df.groupby([column_to_count])[column_groupby].count().reset_index()
    rescuer_count.rename(columns={rescuer_count.columns[0]: column_to_count}, inplace=True)
    rescuer_count.columns = [column_to_count, column_to_count+'_COUNT']
    df = df.merge(rescuer_count, how='left', on=column_to_count)
    return df

def label_encoding(df,columns_to_encode):
    labelencoder = LabelEncoder()
    categ_cols = columns_to_encode
    for columns_ in categ_cols:
        df[columns_+'_ENCODED'] = labelencoder.fit_transform(df[columns_].values.astype(str))
    return df

def binarie_fill(df,column):
    df[column] = df[column].fillna(0)
    if True in df[column].tolist():
        df[column]= np.where(df[column]==True,1,0)
    else:
        df[column]= np.where(df[column]==0,0,1)
    return df

"""
TEXT
"""

def apply_tfidf_vectorizer(df, column):
    df[column] = df[column].fillna("missing")
    df[column] = df[column].astype(str)
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), stop_words = stop_words_fr, lowercase=True, 
                                     max_features=50, binary=True, norm=None,use_idf=False)
    tfidf = vectorizer.fit_transform(df[column])
    tfidf_cols = vectorizer.get_feature_names()
    tmp = pd.DataFrame(data=tfidf.toarray(), columns=['tfidf_' + column + '_' + i for i in tfidf_cols])
    df = pd.concat([df, tmp], axis=1,sort=False)
    return df


def tfidf_nmf_svd(df,text_columns):
    for col_ in tqdm(text_columns):
        text = df[col_].values.tolist()
        cvec = CountVectorizer(min_df=2, ngram_range=(1, 3), max_features=1000,
                               strip_accents='unicode',
                               lowercase=True, analyzer='word', token_pattern=r'\w+',
                               stop_words=stop_words_fr)
        text = [str(element) for element in text]
        cvec.fit(text)
        X = cvec.transform(text)
        df['cvec_sum'] = X.sum(axis=1)
        df['cvec_mean'] = X.mean(axis=1)
        df['cvec_len'] = (X != 0).sum(axis=1)
        tfv = TfidfVectorizer(min_df=2, max_features=200,
                              strip_accents='unicode', analyzer='word',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                              stop_words=stop_words_fr)

        # Fit TFIDF
        X = tfv.fit_transform(text)
        df['tfidf_sum'] = X.sum(axis=1)
        df['tfidf_mean'] = X.mean(axis=1)
        df['tfidf_len'] = (X != 0).sum(axis=1)
        
        """
        n_components = 20

        print('[INFO] Start NMF')

        nmf_ = NMF(n_components=n_components)
        X_nmf = nmf_.fit_transform(X)
        X_nmf = pd.DataFrame(X_nmf, columns=['{}_nmf_{}'.format(col_, i) for i in range(n_components)])
        X_nmf['id'] = df.id.values.tolist()
        df = pd.concat([df.set_index('id'), X_nmf.set_index('id')], sort=False, axis=1).reset_index()
        df.rename(columns={df.columns[0]: 'id'}, inplace=True)

        print('[INFO] Start SVD')
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(X)
        print('fit done')
        X_svd = svd.transform(X)
        X_svd = pd.DataFrame(X_svd, columns=['{}_svd_{}'.format(col_, i) for i in range(n_components)])
        X_svd['id'] = df.id.values.tolist()
        df = pd.concat([df.set_index('id'), X_svd.set_index('id')], sort=False, axis=1).reset_index()
        df.rename(columns={df.columns[0]: 'id'}, inplace=True)
        df.drop(col_, axis=1, inplace=True)
        """
    return df

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
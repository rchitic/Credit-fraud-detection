import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score, confusion_matrix
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import prince
from sklearn.feature_selection import SelectKBest, chi2  # for chi-squared feature selection

# Load data
def load_data(filename):
    # read data and drop unnecessary columns
    data = pd.read_csv(filename)
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop('sk_id_curr', axis=1)
    data = data.drop('code_gender_XNA', axis=1)

    # target dummies -> single target column
    data.loc[data['target_0'] == 1, 'target'] = 0
    data.loc[data['target_0'] == 0, 'target'] = 1
    data.drop(['target_0', 'target_1'], axis=1, inplace=True)

    # separate data into features/label and separate features into numerical/categorical variables
    x = data.drop('target', axis=1)
    y = data['target']

    # separate into numerical and categorical features
    num = x.select_dtypes('float')
    cat = x.select_dtypes('int64')

    # normalize numerical features
    scaler = StandardScaler()
    num = scaler.fit_transform(num)
    num = pd.DataFrame(num)

    # Separate fraud and non-fraud
    features = pd.concat([num, cat], axis=1)
    data0 = features[data['target'] == 0]
    data1 = features[data['target'] == 1]

# Reduce data
def reduce_data(num, cat, pca_flag, mca_flag, km_flag):
    features = pd.DataFrame()
    if pca_flag == 1:
        if mca_flag == 0:
            pca_data = pd.concat([num, cat], axis=1)
        else:
            # MCA
            pca_data = num
            mca = prince.MCA(n_components=89)
            mca = mca.fit(cat)
            mca_output = mca.transform(cat)

            mca_cols = []
            for i in range(89):
                mca_cols.append('MCAv' + str(i))
            mca_output.columns = mca_cols
            features = pd.concat([features, mca_output])

        # PCA
        pca = PCA(n_components=0.95)
        principalComponents = pca.fit(pca_data)
        principalComponents.explained_variance_
        principalComponents = pca.transform(pca_data)

        pca_cols = []
        for i in range(principalComponents.shape[1]):
            pca_cols.append('PCAv' + str(i))
        pca_output = pd.DataFrame(principalComponents, columns=pca_cols)
        features = pd.concat([features, pca_output])
    else:
        features = pd.concat([num, cat], axis=1)

    # Separate fraud and non-fraud
    data0 = features[y == 0]
    data1 = features[y == 1]

    cluster_frac = []
    if km_flag == 1:
        # Cluster non-fraud data
        data0_arr = data0.values
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(data0_arr)
        y_kmeans = kmeans.predict(data0_arr)

        y_kmeans = pd.Series(y_kmeans)
        cluster_counts = y_kmeans.value_counts()
        print('Cluster counts: ', cluster_counts)
        cluster_frac = [cluster_counts[0] / len(y_kmeans), cluster_counts[1] / len(y_kmeans),
                        cluster_counts[2] / len(y_kmeans)]

        data0['cluster'] = y_kmeans

    return data0, data1, cluster_frac

# Create model
def make_model(metr):
    model = keras.Sequential([
        keras.layers.Dense(200, activation='relu', input_shape=(train_features.shape[-1],)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metr)

    return model

# Plot confusion matrix
def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    print('True Negatives: ', cm[0][0])
    print('False Positives): ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])


# Find best upsampling/downsampling factor
def find_k(data0, data1, K, cluster_frac, ep, bsize, metr, results_models_filename):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    auc_dict = {}
    for k in K:
        print(k)
        sample_length = k * len(data1)

        if (cluster_frac) > 0:  # non-fraud data is clustered -> sample proportionally from clusters
            sample0 = data0.loc[data0['cluster'] == 0].sample(int(round(cluster_frac[0] * sample_length)))
            sample1 = data0.loc[data0['cluster'] == 1].sample(int(round(cluster_frac[1] * sample_length)))
            sample2 = data0.loc[data0['cluster'] == 2].sample(int(round(cluster_frac[2] * sample_length)))
            print('Length of samples from clusters 0,1,2: ', len(sample0), len(sample1), len(sample2))

            data0_sample = pd.concat([sample0, sample1, sample2], axis=0).sample(frac=1)
            data0_sample.drop('cluster', axis=1, inplace=True)

        else:  # non-fraud data not clustered -> sample randomly
            data0_sample = data0.sample(sample_length)
        print('Lengths of non-fraud sample and fraud data: ', len(data0_sample), k * len(data1))

        data0_sample['target'] = 0
        data1['target'] = 1

        # Concatenate downsampled non-fraud and oversampled fraud data
        data = data0_sample
        for i in range(1, k + 1):
            data = pd.concat([data, data1], axis=0).sample(frac=1)
        data.head()

        # Train-val-test split
        train_data, test_data = train_test_split(data, test_size=0.2)
        train_data, val_data = train_test_split(train_data, test_size=0.2)

        train_labels = train_data['target']
        test_labels = test_data['target']
        val_labels = val_data['target']
        bool_train_labels = train_labels != 0

        train_features = train_data.drop('target', axis=1)
        test_features = test_data.drop('target', axis=1)
        val_features = val_data.drop('target', axis=1)

        print('Train features & labels: ', train_features.shape, train_labels.shape)
        print('Test features & labels: ', test_features.shape, test_labels.shape)
        print('Validation features & labels: ', val_features.shape, val_labels.shape)

        # Train model
        model = make_model(metr)
        model.fit(train_features, train_labels, validation_data=(val_features, val_labels), epochs=ep, batch_size=bsize)
        model.save(results_models_filename + str(k) + '.h5')

        # Evaluate model
        train_predictions = model.predict(train_features, batch_size=bsize)
        test_predictions = model.predict(test_features, batch_size=bsize)
        results = model.evaluate(test_features, test_labels, batch_size=bsize, verbose=0)
        for name, value in zip(model.metrics_names, results):
            if name == 'auc': auc_dict[k] = value

        # Plot results
        plot_cm(test_labels, test_predictions)
        fp, tp, _ = sklearn.metrics.roc_curve(train_labels, train_predictions)
        plt.plot(100 * fp, 100 * tp, label='train', linewidth=2)
        fp, tp, _ = sklearn.metrics.roc_curve(test_labels, test_predictions)
        plt.plot(100 * fp, 100 * tp, label='test', linewidth=2)
        plt.title('ROC curve k=' + str(k))

    return auc_dict


# Main
# Define args
pca_flag, mca_flag, km_flag = 0, 0, 0
K, ep, bsize = range(1, 12), 200, 128
metr = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]
data_filename = 'data/cleaned_application_data.csv'
results_auc_filename = 'results/AUCs/pca' + str(pca_flag) + '_mca' + str(mca_flag) + '_km' + str(km_flag) + '.npy'
results_models_filename = 'results/models/pca' + str(pca_flag) + '_mca' + str(mca_flag) + '_km' + str(km_flag) + '_k'

# Function calls
num, cat, y = load_data(data_filename)
data0, data1, cluster_frac = reduce_data(num, cat, pca_flag, mca_flag, km_flag)
AUCs = find_k(data0, data1, K, cluster_frac, ep, bsize, metr, results_models_filename)
np.save(results_auc_filename, np.array(list(AUCs)))

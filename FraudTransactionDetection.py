import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set_style('white grid')

np.random.seed(42)

df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.head()

df.shape

print(len(df[df['Amount'] == 0]))

df = df.drop(df[df['Amount'] == 0].index)

df.shape

# Custom colors for data

gray_color = "#CCCCCC" # Grey for regular txs
red_color = "#F0544F" # Red for fraudulent txs
blue_color = "#8CA6F2" # Blue for heatmap
gold_color = '#F2D98C' # Gold for heatmap
green_color = '#A4F28C'# Green for confusion matrix
white_color = '#FFFFFF' # White for confusion matrix

color_pal = [gray_color, red_color]

fig, ax = plt.subplots(ncols=2, figsize=(10,4))

sns.boxplot(data=df,
            x="Class",
            y="Amount",
            hue="Class",
            palette=color_pal,
            showfliers=True,
            ax=ax[0])

sns.boxplot(data=df,
            x="Class",
            y="Amount",
            hue="Class",
            palette=color_pal,
            showfliers=False,
            ax=ax[1])

# Add titles to the plots
ax[0].set_title("Transaction Amount Box Plot (Including Fliers)")
ax[1].set_title("Transaction Amount Box Plot (Excluding Fliers)")

# Update legend labels
legend_labels = ['Non-fraud', 'Fraud']
for i in range(2):
    handles, _ = ax[i].get_legend_handles_labels()
    ax[i].legend(handles, legend_labels)


plt.show()

f, ax = plt.subplots(figsize=(10, 4))

sns.scatterplot(data=df.loc[df.Class==0],
                x='Time',
                y='Amount',
                color=gray_color,
                s=30,
                alpha=1,
                linewidth=0)

ax.set(xlabel=None, xticklabels=[])
plt.ylim(0, 3000)


sns.scatterplot(data=df.loc[df.Class==1],
                x='Time',
                y='Amount',
                color=red_color,
                s=30,
                alpha=1,
                linewidth=0)
plt.ylim(0, 3000)

# Add title to the plot
ax.set_title("Transaction Amount Distribution Over Time")

plt.show()

var = df.columns.values

t0 = df.loc[df['Class'] == 0]
t1 = df.loc[df['Class'] == 1]

num_features = len(var)
num_rows = num_features // 4 + int(num_features % 4 != 0)

fig, ax = plt.subplots(nrows=num_rows, ncols=4, figsize=(16, 28))

for idx, feature in enumerate(var):
    row = idx // 4
    col = idx % 4

    sns.kdeplot(t0[feature], bw_method=0.5, label="Class = 0", color=gray_color, fill=True, warn_singular=False,
                ax=ax[row, col])
    sns.kdeplot(t1[feature], bw_method=0.5, label="Class = 1", color=red_color, warn_singular=False, ax=ax[row, col])

    ax[row, col].set_xlabel(feature, fontsize=12)
    ax[row, col].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()

from matplotlib.colors import LinearSegmentedColormap
f, ax = plt.subplots(figsize = (10, 10))

colors = [gold_color, "#ffffff", blue_color]
color_positions = [1, 0, -1]

color_map = LinearSegmentedColormap.from_list("Custom", colors, N=256, gamma=1.0)

sns.heatmap(
    df.corr('pearson'),
    cmap=color_map,
    square=True,
    center=0,
    cbar_kws={'shrink': .8},
    ax=ax,
    annot=False,
    linewidths=0.1, vmax=1.0, linecolor='white',
)

plt.title('Pearson Correlation of Features', y=1.05, size=15)
plt.show()

from sklearn.tree import DecisionTreeClassifier


def create_dtree(prob=False):
    '''
    This function returns a decision tree classifer with hyperparameters defined as below.
    '''
    classifier = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=None,
        random_state=42)

    def predict(X):
        if prob:
            return classifier.predict_proba(X)[:, 1]  # Return probabilities of the positive class
        else:
            return classifier.predict(X)

    # Allow the classifier to return probabilities, allowing for evaluation metrics such as ROC-AUC to work properly
    classifier.custom_predict = predict
    return classifier

    return classifier

non_pca_features = ['Class','Time','Amount']
pca_features_df = df.drop(columns=non_pca_features)
non_pca_features_df = df[non_pca_features]
y = df['Class']

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Perform a kfold cross validation for better accuracy
skf = StratifiedKFold(n_splits=5)

dtc = create_dtree()

pipeline = Pipeline(
                    [('pca', PCA()),
                     ('dtc', dtc)
])

parameters = {}
parameters['pca__n_components'] = [i for i in range(1, 28)]

gs = GridSearchCV(pipeline, parameters, scoring = 'average_precision', cv=skf, n_jobs=-1)
gs.fit(pca_features_df, y)

pca_gs_results = pd.DataFrame()
pca_gs_results['nComponents'] = [i for i in range(1, 28)]
pca_gs_results.set_index('nComponents', drop=False, inplace=True)
pca_gs_results['Avg Precision'] = gs.cv_results_['mean_test_score']
pca_gs_results['Rank'] = gs.cv_results_['rank_test_score']
pca_gs_results['AvgTime'] = gs.cv_results_['mean_fit_time']

feature_names = ['Avg Precision', 'Rank', 'AvgTime']

pca_gs_results[feature_names]

f, ax = plt.subplots(figsize=(10,5))

ax.axvline(x=16, ymin=0, ymax=1, c="#cccccc", ls='--', linewidth=1.5)

color_map = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)

sns.scatterplot(data=pca_gs_results, x='nComponents', y='Avg Precision', hue='Rank',  palette=color_map, size='AvgTime', ax=ax)



plt.ylim(0.3, 0.85)
plt.legend(loc='lower right')
plt.title('Number of PCA Components and Average Precision Score', y=1.05, size=15)

plt.show()

from sklearn.decomposition import PCA

non_pca_features = ['Time','Amount', 'Class']
pca_features_df = df.drop(columns=non_pca_features)
non_pca_features_df = df[non_pca_features]
y = df['Class']

n_pca_features = 16

pca = PCA(n_components=n_pca_features)
pca.fit(pca_features_df)
pca_df = pd.DataFrame(pca.transform(pca_features_df))
pca_df.columns = [f'PC{i}' for i in range(1, n_pca_features+1)]


pca_df[non_pca_features] = df[non_pca_features]

pca_df.info()

pca_df = pca_df.dropna()

feature_dtype_map = {
    'PC1': 'float32',
    'PC2': 'float32',
    'PC3': 'float32',
    'PC4': 'float32',
    'PC5': 'float32',
    'PC6': 'float32',
    'PC7': 'float32',
    'PC8': 'float32',
    'PC9': 'float32',
    'PC10': 'float32',
    'PC11': 'float32',
    'PC12': 'float32',
    'PC13': 'float32',
    'PC14': 'float32',
    'PC15': 'float32',
    'PC16': 'float32',
    'Time': 'float32',
    'Amount': 'float16',
    'Class': 'uint8',
}

pca_df = pca_df.astype(feature_dtype_map)
pca_df.info()

from matplotlib.colors import LinearSegmentedColormap
f, ax = plt.subplots(figsize = (10, 10))

colors = [gold_color, "#ffffff", blue_color]
color_positions = [1, 0, -1]

color_map = LinearSegmentedColormap.from_list("Custom", colors, N=256, gamma=1.0)

sns.heatmap(
    pca_df.corr('pearson'),
    cmap=color_map,
    square=True,
    center=0,
    cbar_kws={'shrink': .8},
    ax=ax,
    annot=False,
    linewidths=0.1, vmax=1.0, linecolor='white',
)

plt.title('Pearson Correlation of Features', y=1.05, size=15)
plt.show()

pca_df.head()

train_pca_df, test_pca_df, train_pca_labels, test_pca_labels = train_test_split(pca_df.drop('Class', axis=1), pca_df['Class'], test_size=0.25, random_state=42)

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)

# sample for train set
train_pca_df_u, train_pca_labels_u = rus.fit_resample(train_pca_df, train_pca_labels)

# sample from test set
test_pca_df_u, test_pca_labels_u = rus.fit_resample(test_pca_df, test_pca_labels)

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef

# 建立MLP模型
input_dim = len(train_pca_df_u.columns)
clf = MLPClassifier(hidden_layer_sizes=(50,30,20,3), activation='relu', solver='adam', max_iter=500, random_state=42)

# 训练模型
clf.fit(train_pca_df_u, train_pca_labels_u)

# 在测试集上进行预测
test_pca_pred = clf.predict(test_pca_df_u)

# 输出precision、recall、accuracy和MCC
print('Precision:', precision_score(test_pca_labels_u, test_pca_pred))
print('Recall:', recall_score(test_pca_labels_u, test_pca_pred))
print('Accuracy:', accuracy_score(test_pca_labels_u, test_pca_pred))
print('MCC:', matthews_corrcoef(test_pca_labels_u, test_pca_pred))

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

# sample for train set
train_pca_df_o, train_pca_labels_o = sm.fit_resample(train_pca_df, train_pca_labels)

# sample from test set
test_pca_df_o, test_pca_labels_o = sm.fit_resample(test_pca_df, test_pca_labels)
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef

# build MLP model
input_dim = len(train_pca_df_u.columns)
clf = MLPClassifier(hidden_layer_sizes=(40,20,10,5), activation='relu', solver='adam', max_iter=500, random_state=42)

# train the model
clf.fit(train_pca_df_o, train_pca_labels_o)

# predict on the test
test_pca_pred = clf.predict(test_pca_df_o)

# output precision、recall、accuracy and MCC
print('Precision:', precision_score(test_pca_labels_o, test_pca_pred))
print('Recall:', recall_score(test_pca_labels_o, test_pca_pred))
print('Accuracy:', accuracy_score(test_pca_labels_o, test_pca_pred))
print('MCC:', matthews_corrcoef(test_pca_labels_o, test_pca_pred))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras.layers import Dropout
# split label from dataset
X = df.drop('Class', axis=1).values
y = df['Class'].values
# split test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

latent_dim = 2


class Sampling(layers.Layer):
    """用于对z进行采样的类"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = Sequential([
            layers.InputLayer(input_shape=(30,)),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(2 * self.latent_dim),
        ])

        self.decoder = Sequential([
            layers.InputLayer(input_shape=(self.latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(30, activation='sigmoid'),
        ])

    def encode(self, x):
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, log_var

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mean, log_var):
        z = Sampling()([mean, log_var])
        return z

    def call(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)

        # compute KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1))

        # add KL divergence loss to total loss of the model
        self.add_loss(kl_loss)

        return reconstructed

    vae = VAE(latent_dim=2)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(X_train, X_train, epochs=20, batch_size=64, shuffle=True)
    clf = MLPClassifier(hidden_layer_sizes=(60, 30, 20, 5), activation='relu', solver='adam', max_iter=500,
                        random_state=42)

    # train the model
    clf.fit(X_train, y_train)

    # predict on the test
    test_pred = clf.predict(X_test)

    # output precision、recall、accuracy and MCC
    print('Precision:', precision_score(y_test, test_pred))
    print('Recall:', recall_score(y_test, test_pred))
    print('Accuracy:', accuracy_score(y_test, test_pred))
    print('MCC:', matthews_corrcoef(y_test, test_pred))
    
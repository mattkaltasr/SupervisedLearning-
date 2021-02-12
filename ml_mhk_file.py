import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve

# define data filenames
data_filenames = [
    'winequality-red.csv',
    'winequality-white.csv',
    'abalone.csv',
]

'''
There are 2 CSV files with the same columns.
Both CSV files uses ';' as the separator.
The datasets are related to red and white wine.
'''

# read the 1st data file: red wine dataset
# data_red = pd.read_csv(data_filenames[0], sep=';')

# add color attribute
'''
assume:
	0 is white wine
	1 is red wine
'''
# data_red['color'] = 1

# read the 2nd data file: white wine dataset
# data_white = pd.read_csv(data_filenames[1], sep=';')
# read data for abalone
data_alb = pd.read_csv(data_filenames[2], sep=',')
# shape =data_alb.shape
# # dataframe.size
# size = data_alb.size
# # printing size and shape
# print("Size = {}\nShape ={}\nShape[0] x Shape[1] = {}".
# format(size, shape, shape[0]*shape[1]))
# exit()

# add color attribute
# data_white['color'] = 0

# concatenate dataframes
data = data_alb

# data visualization
# distribution of the target attribute: quality

labels, counts = np.unique(data['Rings'], return_counts=True)
plt.bar(labels, counts, align='center')
plt.xlabel('Rings')
plt.ylabel('Count')
plt.title('ring Distribution')
plt.show()

# Let's check dependecy of the quality on other attributes
for field in data.drop('Rings', axis=1):
    # plt.bar(
    #	x=data['quality'],
    #	height=[1,2,3,4,5,6],#data[field],
    # edgecolor='white',
    # )
    sns.barplot(x='Rings', y=field, data=data)
    plt.xlabel('Rings')
    plt.ylabel(field.capitalize())
    # plt.title(f'Quality Distribution')

    plt.show()

'''
Based on 'quality' distribution, we can arrange quality attribute in three groups:
low: 3-4, normal: 5-6, high: 7-9. Let's map this values as follow:
0: low
1: normal
2: high 
As we can see, 'density' and 'pH' do not give any specification to classify the quality.
So, we can drop these atributes.
'''

# # redefine 'quality' attribute
# data['quality'] = data['quality'].map({
#     3: 0,
#     4: 0,
#     5: 1,
#     6: 1,
#     7: 2,
#     8: 2,
#     9: 2,
# })
# seperate the alb dataframe
y = data_alb['Rings']
data_alb = data_alb.drop(['Rings'], axis=1, )

# # separate the dataframe
# y = data['quality']
# data = data.drop([
#     'quality',
#     'density',
#     'pH',
# ],
#     axis=1,
# )
# # Split dataframe into train and test subsets
# x_trainA, x_testA, y_trainA, y_testA = train_test_split(
#     data_alb,
#     y2,
#     test_size=0.33,
# )
'''to do  get all the plots for alb and run the tree ML programs for it 
'''
# Split dataframe into train and test subsets
x_train, x_test, y_train, y_test = train_test_split(
    data,
    y,
    test_size=0.33,
)

# Prediction Models
# store the scores of the models
scores_train = {}
scores_test = {}

# Decision Tree wine
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)  # train model
scores_test['DT'] = dtree.score(x_test, y_test)  # score model

# Neural Networks
# Multi-Layer Perceptron Classifier
mlp = MLPClassifier()
mlp.fit(x_train, y_train)  # train model
scores_test['MLP'] = mlp.score(x_test, y_test)  # score model

# Boosted Decision Tree
# Gradient Boosting Classifier
gboost = GradientBoostingClassifier()
gboost.fit(x_train, y_train)  # train model
scores_test['Boosting'] = gboost.score(x_test, y_test)  # score model

# Support Vector Machines
svm = SVC()
svm.fit(x_train, y_train)  # train model
scores_test['SVM'] = svm.score(x_test, y_test)  # score model

# k-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)  # train model
scores_test['k-NN'] = knn.score(x_test, y_test)  # score model

# output scores
print(scores_test)

train_sizes, train_scores, test_scores = learning_curve(dtree, x_train, y_train, n_jobs=-1, cv=5,
                                                        train_sizes=np.linspace(.1, 1.0, 5),
                                                        verbose=0)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("DTlassifier")
plt.legend(loc="best")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.gca().invert_yaxis()
# box-like grid
plt.grid()

# plot the std deviation as a transparent range at each training set size
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                 color="g")

# plot the average training and test score lines at each training set size
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

# sizes the window for readability and displays the plot
# shows error from 0 to 1.1
plt.ylim(-.1, 1.1)
plt.show()

# plot scores
plt.bar(
    x=range(len(scores_test)),
    height=list(scores_test.values()),
    align='center',
)
plt.xticks(range(len(scores_test)), list(scores_test.keys()))
plt.ylabel('Confidence Score')

plt.show()


# plot learning curve
def plot_learning_curve(estimator, estimator_title):
    fig, axes = plt.subplots(2, 1, figsize=(15, 15))
    y_lim = (0.6, 1.01)
    title = f"Learning Curves {estimator_title}"
    train_sizes, train_scores, test_scores, _, _ = \
        learning_curve(estimator, data, y,
                       cv=10,
                       return_times=True, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    axes[0].set_title(title)
    axes[0].set_ylim(*y_lim)
    axes[0].set_xlabel("Training size")
    axes[0].set_ylabel("Accuracy")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="g",
                 label="Training score")
    axes[0].legend(loc="best")
    axes[1].set_xlabel("Training size")
    axes[1].set_ylabel("Accuracy")
    axes[1].plot(train_sizes, test_scores_mean, 'o-', color="b",
                 label="Test score")
    axes[1].legend(loc="best")

    plt.show()


plot_learning_curve(dtree, "Decision Tree Classifier")
plot_learning_curve(mlp, "MLP Classifier")
plot_learning_curve(svm, "SVM Classifier")
plot_learning_curve(gboost, "Gradient Boosting Classifier")
plot_learning_curve(knn, "K-Neighbors Classifier")
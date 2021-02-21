import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
import time
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
scores_train_times = {}
scores_test = {}

# Decision Tree wine
dtree = DecisionTreeClassifier()
start = time.time()
dtree.fit(x_train, y_train)  # train model
stop = time.time()
print(f" Dtree Training time: {stop - start}s")
scores_train_times['DT'] =  {stop - start}
scores_test['DT'] = dtree.score(x_test, y_test)  # score model
####   below sample of trees depth and acuracy 1-30
max_depth = []
acc_gini = []
acc_entropy = []
for i in range(1, 30):
    dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)
    dtree.fit(x_train, y_train)
    pred = dtree.predict(x_test)
    acc_gini.append(accuracy_score(y_test, pred))
    dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)
    dtree.fit(x_train, y_train)
    pred = dtree.predict(x_test)
    acc_entropy.append(accuracy_score(y_test, pred))
    max_depth.append(i)

d = pd.DataFrame({'acc_gini' :pd.Series(acc_gini),
'acc_entropy':pd.Series(acc_entropy),
 'max_depth':pd.Series(max_depth)})
# visualizing changes in parameters


# print(d)
plt.plot(max_depth, acc_gini, data=None, label='gini')
plt.plot(max_depth, acc_entropy, data=None, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()
plt.show()
# print(max_depth)
# print(acc_gini)
# print(acc_entropy)
# exit()

# Neural Networks
# Multi-Layer Perceptron Classifier
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8),max_iter=300)
start = time.time()
mlp.fit(x_train, y_train)  # train model
stop = time.time()
print(f"NN Training time: {stop - start}s")
scores_train_times['NN'] =  {stop - start}
scores_test['MLP'] = mlp.score(x_test, y_test)  # score model

# Boosted Decision Tree
# Gradient Boosting Classifier
gboost = GradientBoostingClassifier()
start = time.time()
gboost.fit(x_train, y_train)  # train model
stop = time.time()
print(f" Boost Training time: {stop - start}s")
scores_train_times['Boost'] ={stop - start}
scores_test['Boosting'] = gboost.score(x_test, y_test)  # score model

# Support Vector Machines
svm = SVC()
start = time.time()
svm.fit(x_train, y_train)  # train model
stop = time.time()
print(f" svc Training time: {stop - start}s")
scores_train_times['sv'] =  {stop - start}
scores_test['SVM'] = svm.score(x_test, y_test)  # score model

# k-Nearest Neighbors   # default is 5
knn = KNeighborsClassifier(n_neighbors=5)
start = time.time()
knn.fit(x_train, y_train)  # train model
stop = time.time()
print(f" knn Training time: {stop - start}s")
scores_train_times['knn'] =  {stop - start}
scores_test['k-NN'] = knn.score(x_test, y_test)  # score model

# k-Nearest Neighbors   # 3
knn_3 = KNeighborsClassifier(n_neighbors=3)
start = time.time()
knn_3.fit(x_train, y_train)  # train model
stop = time.time()
print(f" knn_3 Training time: {stop - start}s")
scores_train_times['knn_3'] =  {stop - start}
scores_test['k-NN_3'] = knn_3.score(x_test, y_test)  # score model

# k-Nearest Neighbors   # 10
knn_10 = KNeighborsClassifier(n_neighbors=10)
start = time.time()
knn_10.fit(x_train, y_train)  # train model
stop = time.time()
print(f" knn_10 Training time: {stop - start}s")
scores_train_times['knn_10'] =  {stop - start}
scores_test['k-NN_10'] = knn_3.score(x_test, y_test)  # score model


# output scores
print(scores_test)
# print(scores_train_times)

###################  DTREE ######################################################
# comment out for different depth
# dtree = DecisionTreeClassifier(max_depth=7)
start = time.time()
train_sizes, train_scores, test_scores = learning_curve(dtree, x_train, y_train, n_jobs=-1, cv=5,
                                                        train_sizes=np.linspace(.1, 1.0, 5),
                                                        verbose=0)
stop = time.time()
print(f" DT CV Classifier: {stop - start}s")
scores_train_times['DT CV Classifier'] = {stop - start}
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

################################################################################
# exit()
################### KNN ##################
knn_list= {knn,knn_3 ,knn_10}


for i in knn_list:
    train_sizes, train_scores, test_scores = learning_curve(i, x_train, y_train, n_jobs=-1, cv=10,
                                                            train_sizes=np.linspace(.1, 1.0, 5),
                                                            verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()

    plt.title("KNN 5 ,3,10 ")
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



#########################################################
##########  NURAL NETWORK #################

# mlp = MLPClassifier(hidden_layer_sizes=(8,8,8),max_iter=300)
start = time.time()
train_sizes, train_scores, test_scores = learning_curve(mlp, x_train, y_train, n_jobs=-1, cv=10,
                                                        train_sizes=np.linspace(.1, 1.0, 5),
                                                        verbose=0)
stop = time.time()
print(f" NN MLPClassifier: {stop - start}s")
scores_train_times['NN MLPClassifier'] = {stop - start}
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Multi-Layer Perceptron Classifier")
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


##################################################
##########  boost #################
start = time.time()
train_sizes, train_scores, test_scores = learning_curve(gboost, x_train, y_train, n_jobs=-1, cv=10,
                                                        train_sizes=np.linspace(.1, 1.0, 5),
                                                        verbose=0)
stop = time.time()
print(f" GBOOST: {stop - start}s")
scores_train_times['GBOOST'] = {stop - start}
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("GradientBoostingClassifier")
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


##################################################

################### svm ##################
svm_list= {'linear', 'poly', 'rbf', 'sigmoid'}


for i in svm_list:
    start = time.time()
    train_sizes, train_scores, test_scores = learning_curve(SVC(kernel=i), x_train, y_train, n_jobs=-1, cv=10,
                                                            train_sizes=np.linspace(.1, 1.0, 5),
                                                            verbose=0)
    stop = time.time()
    print(f" 'linear', 'poly', 'rbf', 'sigmoid': {stop - start}s")
    scores_train_times[i] = {stop - start}
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()

    plt.title("svm 'linear', 'poly', 'rbf', 'sigmoid' ")
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



#########################################################

# plot scores
plt.bar(
    x=range(len(scores_test)),
    height=list(scores_test.values()),
    align='center',
)
plt.xticks(range(len(scores_test)), list(scores_test.keys()))
plt.ylabel('Confidence Score')

plt.show()
# # plot time
# plt.bar(
#     x=range(len(scores_train_times)),
#     height=list(scores_train_times.values()),
#     align='center',
# )
# plt.xticks(range(len(scores_train_times)), list(scores_train_times.keys()))
# plt.ylabel('run times')
#
# plt.show()

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
plot_learning_curve(knn, "K_5-Neighbors Classifier")
plot_learning_curve(knn_3, "K_3-Neighbors Classifier")
plot_learning_curve(knn_10, "K_10-Neighbors Classifier")
print(scores_train_times)


from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


###---------------------------------------------------------------------------------
### 1st Model
###---------------------------------------------------------------------------------
bayes_model = MultinomialNB(alpha=0.02) 

SGD_model = SGDClassifier(
    max_iter = 8000,
    tol = 1e-4,
    loss = "modified_huber"
)

kNN_model = KNeighborsClassifier(
    n_neighbors = 30,
    metric = 'cosine'
)

p6 = {'n_iter': 2500,'verbose': -1,'objective': 'cross_entropy','metric': 'auc',
    'learning_rate': 0.00581909898961407, 'colsample_bytree': 0.78,
    'colsample_bynode': 0.8, 'lambda_l1': 4.562963348932286, 
    'lambda_l2': 2.97485, 'min_data_in_leaf': 115, 'max_depth': 23, 'max_bin': 898}
LGBM_model = LGBMClassifier(**p6) 

Cat_model = CatBoostClassifier(iterations=2000,
                           verbose=0,
                           l2_leaf_reg=6.6591278779517808,
                           learning_rate=0.005599066836106983,
                           subsample = 0.4,
                           allow_const_label=True,
                           loss_function = 'CrossEntropy')  # 6h

ETR_model = ExtraTreesClassifier(
    n_estimators=100,
    criterion='gini'
)

RF_model = RandomForestClassifier(criterion='entropy')

GPC_model = GaussianProcessClassifier()

SVC_model = SVC(
    probability = True
)

Per_model = Perceptron(
    penalty = 'elasticnet'
)

NC_model = NearestCentroid()

Ctg_model = CategoricalNB(
    alpha = 0.02
)

LR_model = LogisticRegression(
    penalty = "elasticnet",
    solver = "saga",
    max_iter = 500,
    l1_ratio = 0.5
)


###---------------------------------------------------------------------------------
### 2nd Model
###---------------------------------------------------------------------------------

LR_2nd_model = LogisticRegression(
    max_iter = 1000,
    C = 0.01
)

SGD_2nd_model = SGDClassifier(
    loss='log_loss', 
    tol=1e-4, 
    alpha=0.1
)

XGB_model = XGBClassifier(
    objective = 'binary:logistic', 
    eval_metric = 'auc',
    eta = 0.01,
)
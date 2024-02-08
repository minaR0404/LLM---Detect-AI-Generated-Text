
###---------------------------------------------------------------------------------
### library
###---------------------------------------------------------------------------------
import numpy as np
import copy

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier

from config.setting import CFG

from utils.data import train, test, submission
from utils.vectorizer import tf_train, tf_test
from utils.llm import probs_tr, probs


###---------------------------------------------------------------------------------
### Inference
###---------------------------------------------------------------------------------
if __name__ == '__main__':

    if len(test.text.values) <= 5:
        submission.to_csv("submission.csv", index=False)

    else:
        ### 1.Predict estimators
        estimators = CFG.first_model

        skf = StratifiedKFold(
            n_splits = 5,
            shuffle = False
        )

        y = train['label'].values
        fin_train = []
        fin_test = []

        for est in estimators:

            oof_train = np.zeros(tf_train.shape[0])
            oof_test = np.zeros(tf_test.shape[0])
            oof_test_skf = np.zeros((5, tf_test.shape[0]))

            for i, (tr_idx, val_idx) in enumerate(skf.split(tf_train, y)):
                print(f'[CV : {est}] {i+1}/{5}')
                X_train, X_valid = tf_train[tr_idx], tf_train[val_idx]
                y_train, y_valid = y[tr_idx], y[val_idx]
                
                est_cv = copy.deepcopy(est)
                est_cv.fit(X_train, y_train)

                oof_train[val_idx] = est_cv.predict_proba(X_valid)[:,1]
                oof_test_skf[i, :] = est_cv.predict_proba(tf_test)[:,1]

            oof_test[:] = oof_test_skf.mean(axis=0)
            fin_train.append(oof_train)
            fin_test.append(oof_test)
            
        fin_train.append(probs_tr)
        fin_test.append(probs)

        
        final_train = np.stack([fin_train[i] for i in range(len(fin_train))], axis=1)
        final_test = np.stack([fin_test[i] for i in range(len(fin_test))], axis=1)

        
        ### 2.Stacking
        Stack_model = VotingClassifier(
            estimators = CFG.second_model,
            weights = [0.8, 0.2],
            voting = 'soft',
            n_jobs = -1
        )

        Stack_model.fit(final_train, train['label'].values)

        final_preds = Stack_model.predict_proba(final_test)[:,1]

        submission['generated'] = final_preds.astype(np.float16)
        submission.to_csv("submission.csv", index=False)
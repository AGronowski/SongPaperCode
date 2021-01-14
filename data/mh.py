import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import pickle as pkl
from sklearn.model_selection import train_test_split

# used for saving mental_health_dataset
# saved mental health dataset can then be used with the program for the adult dataset
# adult variable must be set to False in data.adult and examples.adult


def save_adult_datasets():
    # header is removed automatically (header=0)
    df = pd.read_csv('imputed_cleaned_mental_health.csv')

    # drop all rows where Gender is other. inplace=True means it's permanently removed from the original df
    # we now have 1402 examples instead of 1428
    df.drop(df[df['Gender'] == "other"].index, inplace=True)

    columns = ['mh_coverage_flag', 'mh_resources_provided', 'mh_anonimity_flag',
                    'mh_prod_impact', 'mh_medical_leave', 'mh_discussion_neg_impact',
                    'mh_family_hist', 'mh_disorder_past', 'AgeBinary', 'Gender','treatment']

    # drop all columns except for feature_cols
    df = df[columns]

    # replace the binary features with 0s and 1s
    df['Gender'] = df['Gender'].replace(to_replace=['male','female'],value=[1,0])
    df['AgeBinary'] = df['AgeBinary'].replace(to_replace=['< 40yo','>= 40yo'],value=[1,0])

    # these are non-binary features that will be one-hot encoded
    feature_cols = ['mh_coverage_flag', 'mh_resources_provided', 'mh_anonimity_flag',
                    'mh_prod_impact', 'mh_medical_leave', 'mh_discussion_neg_impact',
                    'mh_family_hist', 'mh_disorder_past']

    # one-hot encode categorical data
    for feature in feature_cols:
        # use pd.concat to join the new one hot encoded columns with the original dataframe
        df = pd.concat([df,pd.get_dummies(df[feature], prefix=feature)],axis=1)
        # now drop the original column (not needed anymore)
        df.drop([feature],axis=1, inplace=True)

    # label is whether the person sought mental health treatment
    y = df['treatment']
    # x is the features + sensitve variable
    x = df.drop(['treatment'],axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size=0.3)

    # separate out sensitve variable from rest of the features
    u_train = x_train['Gender']
    u_test = x_test['Gender']
    features_train = x_train.drop(['Gender'],axis=1)
    features_test = x_test.drop(['Gender'],axis=1)

    # convert the dataframes to numpy arrays of bools
    mh_binary_train = features_train.values.astype(np.bool), u_train.values.astype(np.bool), y_train.values.astype(np.bool)
    mh_binary_test = features_test.values.astype(np.bool), u_test.values.astype(np.bool), y_test.values.astype(np.bool)

    # export as binary pickle file
    # wb - write binary
    with open('mh_binary_train.pkl', 'wb') as f:
        pkl.dump(mh_binary_train, f)
    with open('mh_binary_test.pkl', 'wb') as f:
        pkl.dump(mh_binary_test, f)

# save_adult_datasets is then used by create_adult_datasets() in data.adult


if __name__ == '__main__':
    save_adult_datasets()
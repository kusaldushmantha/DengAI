import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

feature_train = pd.read_csv('dengue_features_train.csv')
feature_label = pd.read_csv('dengue_labels_train.csv')
feature_test = pd.read_csv('dengue_features_test.csv')

feature_test = feature_test.fillna(method='ffill')

train_set = pd.concat((feature_train,feature_label.iloc[:,3]), axis=1)
train_set = train_set.fillna(method='ffill')
train_set = train_set.drop(['weekofyear','year','week_start_date'], axis=1)

def feature_drop(train, test, method):
    pr = ['city','reanalysis_tdtr_k','ndvi_se','reanalysis_max_air_temp_k',
          'station_max_temp_c', 
          'reanalysis_relative_humidity_percent',
          'ndvi_sw','total_cases']
    mr = ['city','ndvi_se','ndvi_sw','ndvi_nw','reanalysis_tdtr_k',
          'reanalysis_specific_humidity_g_per_kg', 
          'reanalysis_avg_temp_k', 'total_cases']       
            
    features = ['city','reanalysis_tdtr_k', 'ndvi_se', 'ndvi_sw', 'total_cases']
    
    if(method==0):
        res_file = open('res_file.txt','a+')
        res_file.write('method'+str(method))
        res_file.write('\n')
        res_file.flush()
        res_file.close()
        train = train[pr]
        test = test[['city','reanalysis_tdtr_k','ndvi_se','reanalysis_max_air_temp_k',
          'station_max_temp_c', 
          'reanalysis_relative_humidity_percent',
          'ndvi_sw']]
    elif(method==1):
        res_file = open('res_file.txt','a+')
        res_file.write('method'+str(method))
        res_file.write('\n')
        res_file.flush()
        res_file.close()
        train = train[mr]
        test = test[['city','ndvi_se','ndvi_sw','ndvi_nw','reanalysis_tdtr_k',
          'reanalysis_specific_humidity_g_per_kg', 
          'reanalysis_avg_temp_k']]
    else:
        res_file = open('res_file.txt','a+')
        res_file.write('method'+str(method))
        res_file.write('\n')
        res_file.flush()
        res_file.close()
        train = train[features]
        test = test[['city','reanalysis_tdtr_k', 'ndvi_se', 'ndvi_sw']]
    return (train, test)

feature_dropped_set = feature_drop(train=train_set,test=feature_test,method=1)
train_set = feature_dropped_set[0]
feature_test = feature_dropped_set[1]

train_set_sj = train_set[train_set['city']=='sj']
train_set_sj = train_set_sj.drop(['city'], axis=1)
X_set_sj = train_set_sj.iloc[:,:-1].values
y_set_sj = train_set_sj.iloc[:,6].values

X_train_sj, X_test_sj, y_train_sj, y_test_sj = train_test_split(X_set_sj,
                                                                y_set_sj,
                                                                test_size=0.2, 
                                                                random_state=0)
train_set_iq = train_set[train_set['city']=='iq']
train_set_iq = train_set_iq.drop(['city'], axis=1)
X_set_iq = train_set_iq.iloc[:,:-1].values
y_set_iq = train_set_iq.iloc[:,6].values

X_train_iq, X_test_iq, y_train_iq, y_test_iq = train_test_split(X_set_iq,
                                                                y_set_iq,
                                                                test_size=0.2, 
                                                                random_state=0)
        

def evaluate_random_forest_model_sj():
    regressor_sj = RandomForestRegressor(n_estimators=100, 
                                         criterion='mae',
                                         n_jobs=-1,
                                         random_state=0)
    param_grid = {'n_estimators':[100,150,200,250,300,350,400,450,500,550]}
    cv = GridSearchCV(estimator=regressor_sj, param_grid=param_grid, n_jobs=-1)
    cv.fit(X_train_sj, y_train_sj)
    print(cv.best_params_)
    res_file = open('res_file.txt','a+')
    res_file.write('sj best_estimator:'+str(cv.best_params_['n_estimators']))
    res_file.write('\n')
    res_file.flush()
    res_file.close()
    regressor_sj.fit(X_train_sj, y_train_sj)

    return regressor_sj

def evaluate_random_forest_model_iq():
    regressor_iq = RandomForestRegressor(n_estimators=100, 
                                         criterion='mae',
                                         n_jobs=-1,
                                         random_state=0)
    param_grid = {'n_estimators':[100,150,200,250,300,350,400,450,500,550]}
    cv = GridSearchCV(estimator=regressor_sj, param_grid=param_grid, 
                      n_jobs=-1, cv=5)
    cv.fit(X_train_iq, y_train_iq)
    res_file = open('res_file.txt','a+')
    res_file.write('iq best_estimator:'+str(cv.best_params_['n_estimators']))
    res_file.write('\n')
    res_file.flush()
    res_file.close()
    print(cv.best_params_)
    regressor_iq.fit(X_train_iq, y_train_iq)

    return regressor_iq

regressor_sj = evaluate_random_forest_model_sj()
regressor_iq = evaluate_random_forest_model_iq()

def predict_sj(regressor_sj, X_test_sj, y_test_sj):
    y_pred = regressor_sj.predict(X_test_sj)
    mae = mean_absolute_error(y_test_sj, y_pred)
    print('sj mae:'+str(mae))
    res_file = open('res_file.txt','a+')
    res_file.write('sj mae:'+str(mae))
    res_file.write('\n')
    res_file.flush()
    res_file.close()

def predict_iq(regressor_iq, X_test_iq, y_test_iq):
    y_pred = regressor_iq.predict(X_test_iq)
    mae = mean_absolute_error(y_test_iq, y_pred)
    print('iq mae:'+str(mae))
    res_file = open('res_file.txt','a+')
    res_file.write('iq mae:'+str(mae))
    res_file.write('\n')
    res_file.write('\n')
    res_file.flush()
    res_file.close()

predict_sj(regressor_sj=regressor_sj, X_test_sj=X_test_sj, y_test_sj=y_test_sj)
predict_iq(regressor_iq=regressor_iq, X_test_iq=X_test_iq, y_test_iq=y_test_iq)

def predict_results(regressor_sj, regressor_iq, feature_test):
    test_sj = feature_test[feature_test['city']=='sj']
    test_iq = feature_test[feature_test['city']=='iq']
    test_sj = test_sj.drop(['city'], axis=1)
    test_iq = test_iq.drop(['city'], axis=1)
    pred_sj = regressor_sj.predict(test_sj)
    pred_iq = regressor_iq.predict(test_iq)
    result = np.concatenate((pred_sj,pred_iq))
    return result.astype(int)

result = predict_results(regressor_sj=regressor_sj,
                         regressor_iq=regressor_iq,
                         feature_test=feature_test)
    
    
    
    
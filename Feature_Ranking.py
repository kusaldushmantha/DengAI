import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from Pertubation_Rank import pertubation_rank
from sklearn.model_selection import train_test_split

feature_train = pd.read_csv('dengue_features_train.csv')
feature_label = pd.read_csv('dengue_labels_train.csv')
feature_test = pd.read_csv('dengue_features_test.csv')

feature_test = feature_test.fillna(method='ffill')

train_set = pd.concat((feature_train,feature_label.iloc[:,3]), axis=1)
train_set = train_set.fillna(method='ffill')
    
Y = train_set.iloc[:,24].values
train_set = train_set.drop(['city','year','weekofyear',
                            'week_start_date','total_cases'], axis=1)
X = train_set.as_matrix()
col_names = train_set.columns

ranks = {}

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

# Stalibility selection via RandomizedLasso
rlasso = RandomizedLasso(alpha=0.4)
rlasso.fit(X, Y)
ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), col_names)
print('Randomised Lasso finished')

# Recursive feature elimination
lr = LinearRegression(normalize=True)
lr.fit(X, Y)
rfe = RFE(lr, n_features_to_select=1, verbose=3)
rfe.fit(X, Y)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), col_names, order=-1)
print("RFE Finished")

# Linear Model Feature Ranking
# Using Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(X,Y)
ranks["LinReg"] = ranking(np.abs(lr.coef_), col_names)

# Using Ridge 
ridge = Ridge(alpha = 7)
ridge.fit(X,Y)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), col_names)

# Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), col_names)

# Random Forest Feature Ranking
rf = RandomForestRegressor(n_jobs=-1, n_estimators=300, criterion= 'mae')
rf.fit(X,Y)
ranks["RandForest"] = ranking(rf.feature_importances_, col_names);

r = {}
for name in col_names:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)
 
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print("\t%s" % "\t".join(methods))

for name in col_names:
    print("%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods]))))
    
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)

# Pertubation Ranking
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
result = pertubation_rank(model=rf,x=X_test,y=y_test,
                          names=col_names,regression=True)    

# Selected Features

# Pertubation rank
# ['reanalysis_tdtr_k','ndvi_se','reanalysis_max_air_temp_k',
#  'station_max_temp_c', 'reanalysis_relative_humidity_percent','ndvi_sw']

# Mean Rank
# ['ndvi_se','ndvi_sw','ndvi_nw','reanalysis_tdtr_k',
#  'reanalysis_specific_humidity_g_per_kg', 'reanalysis_avg_temp_k']

pr = ['reanalysis_tdtr_k','ndvi_se','reanalysis_max_air_temp_k','station_max_temp_c', 'reanalysis_relative_humidity_percent','ndvi_sw']
mr = ['ndvi_se','ndvi_sw','ndvi_nw','reanalysis_tdtr_k','reanalysis_specific_humidity_g_per_kg', 'reanalysis_avg_temp_k']       
        
features = ['reanalysis_tdtr_k', 'ndvi_se', 'ndvi_sw']

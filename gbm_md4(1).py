import xgboost as xgb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
inicio = datetime.now()

proposicoes = pd.read_csv('v_propos7.csv')
y = proposicoes["aprovado"].values
y = y.reshape(-1, 1)
X = proposicoes.drop("aprovado",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
DM_train = xgb.DMatrix(data = X_train, label = y_train)
DM_test =  xgb.DMatrix(data = X_test,label = y_test)

gbm_param_grid = {
     'colsample_bytree': np.linspace(0.5, 0.9, 5),
     'n_estimators':[20],
     'max_depth': [10]
}

gbm = xgb.XGBRegressor()
#Vamos realizar uma validação cruzada de 10 vezes usando o erro quadrático médio como método de pontuação.
grid_mse = GridSearchCV(estimator = gbm, param_grid = gbm_param_grid, scoring = 'neg_mean_squared_error', cv = 10, verbose = 5)

#Ajustar grid_mse aos dados, obter melhores parâmetros e melhor pontuação (menor RMSE)
grid_mse.fit(X_train, y_train)
print("Best parameters found: ",grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))

#Gravando modelo
joblib.dump(grid_mse ,'proposicoes.pkl')

#Estimar usando os dados de teste
pred = grid_mse.predict(X_test)
predictions = [round(value) for value in pred]

print("Root mean square error for test dataset: {}".format(np.round(np.sqrt(mean_squared_error(y_test, pred)), 2)))
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
cm = confusion_matrix(y_test, predictions)
print(cm)
print(classification_report(y_test, predictions))
print(datetime.now()-inicio)
print(datetime.now()-inicio)
f = sns.heatmap(cm, annot=True, fmt='d')
plt.show()




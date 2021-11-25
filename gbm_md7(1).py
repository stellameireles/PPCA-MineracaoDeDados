import xgboost as xgb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

inicio = datetime.now()

#Ler CSV
proposicao = pd.read_csv('v_propos7.csv')
# Retirar coulunas
y = proposicao["aprovado"].values
del proposicao['aprovado']
X = proposicao 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

DM_train = xgb.DMatrix(data = X_train, label = y_train)
DM_test =  xgb.DMatrix(data = X_test,label = y_test)

#Ajustar grid_mse aos dados, obter melhores parâmetros e melhor pontuação (menor RMSE)
eval_set = [(X_test, y_test)]
grid_mse = xgb.XGBClassifier(use_label_encoder=False)
grid_mse.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

#Gravando modelo
joblib.dump(grid_mse ,'proposicoes.pkl')

#Estimar usando os dados de teste
pred = grid_mse.predict(X_test)
predictions = [value for value in pred]

# Apresentar Estatísticas
print("Root mean square error for test dataset: {}".format(np.round(np.sqrt(mean_squared_error(y_test, pred)), 2)))
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
print('Matriz de Confusão')
cm = confusion_matrix(y_test, predictions)
print(cm)
print(classification_report(y_test, predictions))
print(datetime.now()-inicio)
f = sns.heatmap(cm, annot=True, fmt='d')
plt.show()




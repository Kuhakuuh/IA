import json
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.tree import plot_tree



async def trainDecisionTreeRegressionDefault():

    df = pd.read_csv('files\kc_house_data.csv', encoding='ISO-8859-1')
    #Remover as colunas 
    df = df.drop([ 'id','date', 'yr_built','zipcode','yr_renovated','sqft_living15','sqft_lot15', 'sqft_above','sqft_basement','waterfront', 'condition', 'floors', 'view'], axis=1)
    print(df)

    #Converter para m2
    sqft = 0.092903
    df['sqft_living'] = df['sqft_living'] * sqft
    df.rename(columns={'sqft_living': 'living_m2'}, inplace=True)

    df['sqft_lot'] = df['sqft_lot'] * sqft
    df.rename(columns={'sqft_lot': 'lot_m2'}, inplace=True)

    # definindo as variáveis
    y = df['price'] #Variavel target
    x = df.loc[:, df.columns != "price"]
    print(df)
    # Dados de treino e teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
        
    # imputer para preencher os valores null com a média
    imputer = SimpleImputer(strategy='mean')
    x_train_imputed = imputer.fit_transform(x_train)
    print("----------------------------------------------------")
    print(x)
    #Modelo Random Forest     
    decisionTree = tree.DecisionTreeRegressor()


    #Treino
    decisionTree.fit(x_train_imputed, y_train)

    # Previsão do frame de teste
    y_pred = decisionTree.predict(x_test)

    score = decisionTree.score(x_test, y_test)

    print("Score :",score)

    # Salvar o modelo
    joblib.dump(decisionTree, 'modelo_dt_tuning.zip')

    #Graficos
    importances = decisionTree.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = x.columns
    print(feature_names)

    #variavveis mais importantes
    residuals = y_test - y_pred
    plt.figure()
    plt.title('Feature Importances')
    plt.bar(range(x.shape[1]), importances[indices], align='center')
    plt.xticks(range(x.shape[1]), x.columns[indices], rotation=90)
    plt.tight_layout()
    plot_path = 'plot_decisionTree/plot_importance_tuning.png'
    plt.savefig(plot_path)
    plt.close()

    #Residos
    plt.scatter(y_pred, residuals)
    plt.hlines(0, xmin=min(y_pred), xmax=max(y_pred), colors='r')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plot_path = 'plot_decisionTree/plot_residual_tuning.png'
    plt.savefig(plot_path)
    plt.close()

    #comparaçao entre valores reais e previsões
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Parity Plot')
    #plt.show()
    plot_path = 'plot_decisionTree/plot_parity_tuning.png'
    plt.savefig(plot_path)
    plt.close()



#async def trainDecisionTreeRegressionTuning():

df = pd.read_csv('files\kc_house_data.csv', encoding='ISO-8859-1')
#Remover as colunas 
df = df.drop([ 'id','date', 'yr_built','zipcode','yr_renovated','sqft_living15','sqft_lot15', 'sqft_above','sqft_basement','waterfront', 'condition', 'floors', 'view'], axis=1)
print(df)

#Converter para m2
sqft = 0.092903
df['sqft_living'] = df['sqft_living'] * sqft
df.rename(columns={'sqft_living': 'living_m2'}, inplace=True)

df['sqft_lot'] = df['sqft_lot'] * sqft
df.rename(columns={'sqft_lot': 'lot_m2'}, inplace=True)

# definindo as variáveis
y = df['price'] #Variavel target
x = df.loc[:, df.columns != "price"]
print(df)
# Dados de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    
# imputer para preencher os valores null com a média
imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)
print("----------------------------------------------------")
print(x)
#Modelo Random Forest     
decisionTree = tree.DecisionTreeRegressor(
                                    criterion='squared_error',      
                                    splitter='best',                
                                    max_depth=None,                 # Profundidade máxima da árvore
                                    min_samples_split=2,            # Número mínimo de amostras necessárias para dividir um nó interno
                                    min_samples_leaf=1,             # Número mínimo de amostras necessárias para estar em um nó 
                                    min_weight_fraction_leaf=0.0,   # Fração mínima do peso total das amostras necessária para estar em um nó folha
                                    max_features=None,              # Número máximo de features a serem consideradas para dividir um nó ('None', 'auto', 'sqrt', 'log2' ou um número inteiro)
                                    random_state=42,                
                                    max_leaf_nodes=None,            # Número máximo de folhas
                                    min_impurity_decrease=0.0                                       
                                    )


#Treino
decisionTree.fit(x_train_imputed, y_train)

# Previsão do frame de teste
y_pred = decisionTree.predict(x_test)

score = decisionTree.score(x_test, y_test)

print("Score :",score)

# Salvar o modelo
joblib.dump(decisionTree, 'modelo_dt_tuning.zip')

#Graficos
importances = decisionTree.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = x.columns
print(feature_names)

#variavveis mais importantes
residuals = y_test - y_pred
plt.figure()
plt.title('Feature Importances')
plt.bar(range(x.shape[1]), importances[indices], align='center')
plt.xticks(range(x.shape[1]), x.columns[indices], rotation=90)
plt.tight_layout()
plot_path = 'plot_decisionTree/plot_importance_tuning.png'
plt.savefig(plot_path)
plt.close()

#Residos
plt.scatter(y_pred, residuals)
plt.hlines(0, xmin=min(y_pred), xmax=max(y_pred), colors='r')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plot_path = 'plot_decisionTree/plot_residual_tuning.png'
plt.savefig(plot_path)
plt.close()

#comparaçao entre valores reais e previsões
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Parity Plot')
#plt.show()
plot_path = 'plot_decisionTree/plot_parity_tuning.png'
plt.savefig(plot_path)
plt.close()


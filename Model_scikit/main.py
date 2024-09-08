#%% md
# ### 1.0 Importing Libraries
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%% md
# ### 1.1 Loading the datasets
#%%
archivo_url = 'https://raw.githubusercontent.com/NgoQuocBao1010/Exercise-Correction/main/core/plank_model/train.csv'

archivo = 'train.csv'

ds = pd.read_csv(archivo)

ds
#%%
archivo_url1 = 'https://raw.githubusercontent.com/NgoQuocBao1010/Exercise-Correction/main/core/plank_model/test.csv'

archivo = 'test.csv'

ds_test = pd.read_csv(archivo)

ds_test
#%% md
# ### 1.2 Dataset preprocessing
# Replacing high back and low back with 2 and 1 respectively and correct with 0
#%%
ds_test['label'] = ds_test['label'].replace('H', 2)
ds_test['label'] = ds_test['label'].replace('L', 1)
ds_test['label'] = ds_test['label'].replace('C', 0)
#%%
ds['label'] = ds['label'].replace('H', 2)
ds['label'] = ds['label'].replace('L', 1)
ds['label'] = ds['label'].replace('C', 0)
#%% md
# ### 1.3 Exploratory Data Analysis
#%%
ds.columns
#%%
# dataset correlation matrix
cm = ds.corr()

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, ax=ax, cmap="mako")
plt.show()
#%%
ds["label"].value_counts()
#%%
# plot the distribution of the target variable
sns.histplot(ds["label"], kde=True)
#%% md
# ### 1.4 Data Modeling
#%%
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(ds.loc[:, ds.columns != 'label'], ds["label"].values.ravel())
#%% md
# ### 1.5 Model Evaluation
#%%
# predict the target variable
y_pred = log_model.predict(ds_test.loc[:, ds_test.columns != 'label'])
#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# calculate the accuracy of the model

accuracy = accuracy_score(ds_test['label'], y_pred)
print("Accuracy:", accuracy)

# calculate the confusion matrix
cm = confusion_matrix(ds_test['label'], y_pred)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cm, annot=True, ax=ax)
_ = plt.xlabel("Predicted")
_ = plt.ylabel("Actual")
#%%
from sklearn.metrics import classification_report

print(classification_report(ds_test['label'], y_pred))
#%% md
# ### 1.6 Feature Scaling
#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()), 
    ("logistic", LogisticRegression())
])

pipe.fit(ds.loc[:, ds.columns != 'label'], ds['label'].values.ravel())

y_pred_scale = pipe.predict(ds_test.loc[:, ds_test.columns != 'label'])

accuracy = accuracy_score(ds_test['label'], y_pred_scale)
cm = confusion_matrix(ds_test['label'], y_pred_scale)

print("Accuracy:", accuracy)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cm, annot=True, ax=ax)
_ = plt.xlabel("Predicted")
_ = plt.ylabel("Actual")
#%%
print("Coefficients:", log_model.coef_)
print("Intercept:", log_model.intercept_)
#%% md
# ### 1.7 Feature Importance
#%%
import numpy as np

fig, ax = plt.subplots(figsize=(10, 8))

feature_importances = pd.DataFrame(
    {"column": ds.loc[:, ds.columns != 'label'].columns, "coef": np.abs(pipe.named_steps["logistic"].coef_[0])}
).sort_values(by="coef", ascending=True)

ax.barh(feature_importances["column"], feature_importances["coef"])
#%% md
# ### 1.8 Hyperparameter Tuning
#%%
from sklearn.model_selection import GridSearchCV

param_grid = {
    "logistic__C": [0.1, 1, 10, 100, 1000],
    "logistic__penalty": ["l1", "l2"]
}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(ds.loc[:, ds.columns != 'label'], ds['label'].values.ravel())

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)


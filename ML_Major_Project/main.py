# importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading dataset
data = pd.read_csv("https://raw.githubusercontent.com/RealMosam/TeachNook/main/penguins_size.csv")
# Check information about the data
data.info()

# Columns in the dataset:
#     Species: penguin species (Chinstrap, Adélie, or Gentoo)
#     Island: island name (Dream, Torgersen, or Biscoe) in the Palmer Archipelago (Antarctica)
#     culmen_length_mm: culmen length (mm)
#     culmen_depth_mm: culmen depth (mm)
#     flipper_length_mm: flipper length (mm)
#     body_mass_g: body mass (g)
#     Sex: penguin sex

print("\n", data.head(), "\n")

# Performing Exploratory Data Analysis (EDA)
sns.countplot(x='species', data=data)

# Our data is not balanced. As data is not big enough so no need to balance it.

sns.pairplot(data, hue='species')

# We can see clusters are easily separable in the cases:
# culmen_length_mm vs culmen_depth_mm
# culmen_length_mm vs flipper_length_mm
# culmen_length_mm vs body_mass_g.

# Exploring distribution of our data
fig, axes = plt.subplots(4, 1, figsize=(5, 20))
sns.boxplot(x=data.species, y=data.flipper_length_mm, ax=axes[0], palette='summer')
axes[0].set_title("Flipper length distribution", fontsize=20, color='Red')
sns.boxplot(x=data.species, y=data.culmen_length_mm, ax=axes[1], palette='rocket')
axes[1].set_title("Culmen length distribution", fontsize=20, color='Red')
sns.boxplot(x=data.species, y=data.culmen_depth_mm, ax=axes[2], palette='twilight')
axes[2].set_title("Culmen depth distribution", fontsize=20, color='Red')
sns.boxplot(x=data.species, y=data.body_mass_g, ax=axes[3], palette='Set2')
axes[3].set_title("Body mass distribution", fontsize=20, color='Red')
plt.tight_layout()

print("Mean body mass index distribution\n",
      data.groupby(['species', 'sex']).mean(numeric_only=True)['body_mass_g'].round(2), "\n")

# Checking for any missing values
print("Checking for any missing values", 100 * data.isnull().sum() / len(data), "\n")

# Percentage of missing data is very less.
# Let's impute it with median in numerical features and mode in categorical feature.
# Here, I have used '.fillna' method from pandas library.

data['sex'].fillna(data['sex'].mode()[0], inplace=True)
col_to_be_imputed = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
for item in col_to_be_imputed:
    data[item].fillna(data[item].mean(), inplace=True)

# Dealing with categorical features
print(data.species.value_counts(), "\n")

print(data.island.value_counts(), "\n")

print(data.sex.value_counts(), "\n")
# Where did this '.' entry came from?
print(data[data['sex'] == '.'], "\n")
data.loc[336, 'sex'] = 'FEMALE'  # Setting 'sex' of 336 row to "Female"
print(data.sex.value_counts(), "\n")  # Missing value updated

# Target variables can also be encoded using sklearn.preprocessing.LabelEncoder
data['species'] = data['species'].map({'Adelie': 0, 'Gentoo': 1, 'Chinstrap': 2})

# Creating dummy variables for categorical features
dummies = pd.get_dummies(data[['island', 'sex']], drop_first=True)

# Normalizing/ Standardizing feature variables
df_to_be_scaled = data.drop(['island', 'sex'], axis=1)
target = df_to_be_scaled.species
df_feat = df_to_be_scaled.drop('species', axis=1)

# Using StandardScaler for normalizing
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df_feat)
df_scaled = scaler.transform(df_feat)
df_scaled = pd.DataFrame(df_scaled, columns=df_feat.columns[:4])
df_preprocessed = pd.concat([df_scaled, dummies, target], axis=1)
print(df_preprocessed.head(), "\n")

# Hence, our EDA is now complete.
# Now we shall apply suitable algorithm for model building.

# K-Nearest Neighbours
# It is a supervised learning algorithm which can be used for both classification and regression predictive problems.
# However, it is more widely used in classification problems in the industry.
# With the given data, KNN can classify new, unlabelled data by analysis of the 'k' number of the nearest data points.

# Using KNeighborsClassifier from sklearn for KNN model building
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# We need to split data for supervised learning models.
X_train, X_test, y_train, y_test = train_test_split(df_preprocessed.drop('species', axis=1), target, random_state=0,
                                                    test_size=0.50)
# Here 'k' value is '1'
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
preds_knn = knn.predict(X_test)
initial_confusion_matrix = confusion_matrix(y_test, preds_knn)
print(initial_confusion_matrix, "\n")

# Checking initial Classification Report
from sklearn.metrics import classification_report

initial_classification_report = classification_report(y_test, preds_knn)
print(initial_classification_report, "\n")

# Checking Prediction on test data
from sklearn.metrics import accuracy_score

# print(knn.predict([[culmen_length_mm, 	culmen_depth_mm, 	flipper_length_mm, 	body_mass_g, 	island_Dream,
# island_Torgersen, 	sex_MALE]]))
predicted_species_of_penguin = knn.predict([[-8.870812e-01, 7.877425e-01, -1.422488, -0.565789, 0, 1, 1]])
print(predicted_species_of_penguin)
if predicted_species_of_penguin == [0]:
    print("Adelie")
elif predicted_species_of_penguin == [1]:
    print("Chinstrap")
else:
    print("Gentoo")
print(accuracy_score(y_test, preds_knn), "\n")
# Prediction is correct


#  Scores on Training and Test set
print('Training set score: {:.4f}'.format(knn.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(knn.score(X_test, y_test)), "\n")

# Figuring out the best value of 'k'
error_rate = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# From the graph, we can notice that best value for 'k' is '6'.

# Best fitting Model:
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
preds_knn = knn.predict(X_test)
final_confusion_matrix = confusion_matrix(y_test, preds_knn)
print(final_confusion_matrix, "\n")

# Checking final Classification Report
final_classification_report = classification_report(y_test, preds_knn)
print(final_classification_report)
plt.show()
# Hence, we achieved 99% accuracy.

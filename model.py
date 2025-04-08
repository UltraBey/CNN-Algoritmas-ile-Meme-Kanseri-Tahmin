# model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Veri setini yükleme ve hazırlama
data_file_path = 'breast-cancer-wisconsin.data'

column_names = ['ID', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape', 
                'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 
                'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

data = pd.read_csv(data_file_path, header=None, names=column_names)
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)
data.drop(columns='ID', inplace=True)
data = data.astype(float)

# Veri setinden örnek veriler oluşturma
benign_samples = data[data['Class'] == 2].sample(3, random_state=42).drop(columns='Class').values
malignant_samples = data[data['Class'] == 4].sample(3, random_state=42).drop(columns='Class').values

# Eğitim ve test setlerini ayırma
X = data.drop(columns='Class')
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN modelini eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Test verisi ile sınıf tahminleri yapma
y_pred = knn.predict(X_test)

# Sonuçları değerlendirme
accuracy = accuracy_score(y_test, y_pred) #doğruluk hesaplama
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Doğruluk: {accuracy}')
print(f'Sınıflandırma Raporu:\n{report}')

# Tahmin fonksiyonu
def predict(input_data):
    prediction = knn.predict(input_data)
    return prediction

# Matrix oluşturma fonksiyonu
def plot_confusion_matrix():
    conf_matrix = confusion_matrix(y_test, y_pred)
    return conf_matrix

# Örnek veriler ve predict fonksiyonu app.py dosyasında kullanılacak
example_data = {
    'benign': benign_samples,
    'malignant': malignant_samples,
    'predict': predict,
    'plot_confusion_matrix': plot_confusion_matrix
}

if __name__ == "__main__":
    plot_confusion_matrix()

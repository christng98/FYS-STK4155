#%%
import os
import polars as pl
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.legend_handler import HandlerLine2D
from utils.preprocessing import Preprocessor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, make_scorer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

labels = ['Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'LRTI', 'Pneumonia', 'URTI']

def extractId(filename):
    return filename.split('_')[0]

def customizeDataset(trim_size, data):
    data = data.filter(pl.col('disease') != 'Asthma')
    data = data.filter(pl.col('disease') != 'LRTI')
    
    count = 0
    placeholder = []
    for i in range(len(data)):
        row = data.row(i)
        patient_id, filename, disease = row
        if disease == 'COPD':
            if count < trim_size:
                placeholder.append(row)
                count += 1
        else:
            placeholder.append(row)
    return pl.DataFrame(placeholder, schema=['patient_id', 'filename', 'disease'], orient='row')

def getFeatures(data, feature):
    X_features = []
    for i in range(len(data)):
        soundArr, sample_rate = lb.load(os.getcwd() + '/Project_3/dataset/processed_audio_files/' + data['filename'][i])
        if feature == 'mfcc':
            mfcc = lb.feature.mfcc(y=soundArr, sr=sample_rate, n_mfcc=40).flatten()
            X_features.append(mfcc)
        elif feature == 'cstft':
            cstft = lb.feature.chroma_stft(y=soundArr, sr=sample_rate)
            X_features.append(cstft)
        elif feature == 'mSpec':
            mSpec = lb.feature.melspectrogram(y=soundArr, sr=sample_rate)
            X_features.append(mSpec)
    return X_features

def random_forest(X_train, y_train, X_val, y_val):
    # Performing a grid search to find the optimal max_depth
    # param_grid = {'max_depth': range(1, 30, 2)}
    # rf = RandomForestClassifier(random_state=42)
    # f1_scorer = make_scorer(f1_score, average='weighted')
    # grid_search = GridSearchCV(rf, param_grid, cv=5, scoring=f1_scorer)
    # grid_search.fit(X_train, y_train)
    # best_max_depth = grid_search.best_params_['max_depth']
    # print(f"Best max_depth: {best_max_depth}")
    best_max_depth = 27

    # Train the Random Forest with the best max_depth
    rf = RandomForestClassifier(random_state=42, max_depth=best_max_depth)
    rf.fit(X_train, y_train)

    y_pred_val = rf.predict(X_val)

    # Calculate F1 score, precision, recall, and confusion matrix for validation set
    f1 = f1_score(y_val, y_pred_val, average='weighted')
    precision = precision_score(y_val, y_pred_val, average='weighted')
    recall = recall_score(y_val, y_pred_val, average='weighted')
    conf_matrix = confusion_matrix(y_val, y_pred_val)

    print(f"Validation F1 Score: {f1}")
    print(f"Validation Precision: {precision}")
    print(f"Validation Recall: {recall}")
    print("Validation Confusion Matrix:")
    print(conf_matrix)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=rf.classes_)
    disp.plot()
    plt.show()

    return rf

def decision_tree(X_train, y_train, X_val, y_val):
    # Perform grid search to find the best max_depth
    # param_grid = {'max_depth': range(1, 30, 5), 'criterion': ['gini', 'entropy']}
    # dt = DecisionTreeClassifier(random_state=42)
    # f1_scorer = make_scorer(f1_score, average='weighted')
    # grid_search = GridSearchCV(dt, param_grid, cv=5, scoring=f1_scorer)
    # grid_search.fit(X_train, y_train)
    # best_params = grid_search.best_params_
    # print(f"Best parameters for Decision Tree: {best_params}")

    best_params = {'criterion': 'entropy', 'max_depth': 11}

    # Train the Decision Tree with the best max_depth
    dt = DecisionTreeClassifier(random_state=42, max_depth=best_params['max_depth'], criterion=best_params['criterion'])
    dt.fit(X_train, y_train)

    y_pred_val = dt.predict(X_val)

    # Calculate F1 score, precision, recall, and confusion matrix for validation set
    f1 = f1_score(y_val, y_pred_val, average='weighted')
    precision = precision_score(y_val, y_pred_val, average='weighted')
    recall = recall_score(y_val, y_pred_val, average='weighted')
    conf_matrix = confusion_matrix(y_val, y_pred_val)

    print(f"Validation F1 Score for Decision Tree: {f1}")
    print(f"Validation Precision for Decision Tree: {precision}")
    print(f"Validation Recall for Decision Tree: {recall}")
    print("Validation Confusion Matrix for Decision Tree:")
    print(conf_matrix)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=dt.classes_)
    disp.plot()
    plt.show()

    return dt

def support_vector_machine(X_train, y_train, X_val, y_val):
    # Perform grid search to find the best hyperparameters
    # param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    # svm = SVC(random_state=42)
    # f1_scorer = make_scorer(f1_score, average='weighted')
    # grid_search = GridSearchCV(svm, param_grid, cv=5, scoring=f1_scorer)
    # grid_search.fit(X_train, y_train)
    # best_params = grid_search.best_params_
    # print(f"Best parameters for SVM: {best_params}")

    best_params = {'C': 10, 'kernel': 'linear'}

    # Train the SVM with the best hyperparameters
    svm = SVC(random_state=42, probability=True, C=best_params['C'], kernel=best_params['kernel'])
    svm.fit(X_train, y_train)

    y_pred_val = svm.predict(X_val)

    # Calculate F1 score, precision, recall, and confusion matrix for validation set
    f1 = f1_score(y_val, y_pred_val, average='weighted')
    precision = precision_score(y_val, y_pred_val, average='weighted')
    recall = recall_score(y_val, y_pred_val, average='weighted')
    conf_matrix = confusion_matrix(y_val, y_pred_val)

    print(f"Validation F1 Score for SVM: {f1}")
    print(f"Validation Precision for SVM: {precision}")
    print(f"Validation Recall for SVM: {recall}")
    print("Validation Confusion Matrix for SVM:")
    print(conf_matrix)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=svm.classes_)
    disp.plot()
    plt.show()

    return svm

def logistic_regression(X_train, y_train, X_val, y_val):
    # Train the Logistic Regression with the best hyperparameters
    lr = LogisticRegression(random_state=42, max_iter=500)
    lr.fit(X_train, y_train)

    y_pred_val = lr.predict(X_val)

    # Calculate F1 score, precision, recall, and confusion matrix for validation set
    f1 = f1_score(y_val, y_pred_val, average='weighted')
    precision = precision_score(y_val, y_pred_val, average='weighted')
    recall = recall_score(y_val, y_pred_val, average='weighted')
    conf_matrix = confusion_matrix(y_val, y_pred_val)

    print(f"Validation F1 Score for Logistic Regression: {f1}")
    print(f"Validation Precision for Logistic Regression: {precision}")
    print(f"Validation Recall for Logistic Regression: {recall}")
    print("Validation Confusion Matrix for Logistic Regression:")
    print(conf_matrix)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=lr.classes_)
    disp.plot()
    plt.show()

    return lr
def k_nearest_neighbors(X_train, y_train, X_val, y_val):
    # Perform grid search to find the best hyperparameters
    # param_grid = {'n_neighbors': range(1, 15, 1)}
    # knn = KNeighborsClassifier()
    # f1_scorer = make_scorer(f1_score, average='weighted')
    # grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=f1_scorer)
    # grid_search.fit(X_train, y_train)
    # best_n_neighbors = grid_search.best_params_['n_neighbors']
    # print(f"Best n_neighbors for KNN: {best_n_neighbors}")

    best_n_neighbors = 3
    # Train the KNN with the best hyperparameters
    knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
    knn.fit(X_train, y_train)

    y_pred_val = knn.predict(X_val)

    # Calculate F1 score, precision, recall, and confusion matrix for validation set
    f1 = f1_score(y_val, y_pred_val, average='weighted')
    precision = precision_score(y_val, y_pred_val, average='weighted')
    recall = recall_score(y_val, y_pred_val, average='weighted')
    conf_matrix = confusion_matrix(y_val, y_pred_val)

    print(f"Validation F1 Score for KNN: {f1}")
    print(f"Validation Precision for KNN: {precision}")
    print(f"Validation Recall for KNN: {recall}")
    print("Validation Confusion Matrix for KNN:")
    print(conf_matrix)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=knn.classes_)
    disp.plot()
    plt.show()

    return knn

def main():
    preprocessor = Preprocessor()
    preprocessor.download_data("vbookshelf/respiratory-sound-database")
    preprocessor.load_data()
    preprocessor.df

    # preprocessor.plot_cycle_lengths()
    # preprocessor.plot_cycle_lengths_by_patient()
    # preprocessor.process_audio_files(max_len=6)
    # X_mfcc, X_cstft, X_mSpec = preprocessor.get_features()

    # Create a new dataframe with the class corresponding to the pid for each of the processed audio files
    processed_audio_dir = os.getcwd() + '/Project_3/dataset/processed_audio_files'
    files = os.listdir(processed_audio_dir)

    files_df = pl.DataFrame({
        'patient_id': [extractId(f) for f in files],
        'filename': files
    })

    # print(files_df.head())

    diagnosis_directory = os.path.join(preprocessor.dataset_dir, 'Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv')
    diagnosis = pl.read_csv(diagnosis_directory, has_header=False, new_columns=['patient_id', 'disease'])

    files_df = files_df.with_columns(pl.col('patient_id').cast(pl.Int64))
    data = files_df.join(diagnosis, on='patient_id')

    # disease_counts = data.group_by('disease').len()
    # print(disease_counts)

    # Trimming the amount of COPD patients and removing all asthma patients because it's too few.
    dataset_without_asthma = customizeDataset(600, data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataset_without_asthma, dataset_without_asthma['disease'].to_numpy(), stratify=dataset_without_asthma['disease'].to_numpy(), test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train_mfcc = getFeatures(X_train, 'mfcc')
    X_val_mfcc = getFeatures(X_val, 'mfcc')
    X_test_mfcc = getFeatures(X_test, 'mfcc')

    mean = np.mean(X_train_mfcc, axis=0)
    std = np.std(X_train_mfcc, axis=0)

    X_train_mfcc = (X_train_mfcc - mean) / std
    X_val_mfcc = (X_val_mfcc - mean) / std
    X_test_mfcc = (X_test_mfcc - mean) / std

    X_test_mfcc = np.array(X_test_mfcc)
    X_train_mfcc = np.array(X_train_mfcc)
    X_val_mfcc = np.array(X_val_mfcc)

    smote = SMOTE(random_state=42)
    X_train_mfcc, y_train = smote.fit_resample(X_train_mfcc, y_train)

    encoder = OrdinalEncoder()
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_val = encoder.transform(y_val.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))

    y_train = y_train.flatten()
    y_val = y_val.flatten()
    y_test = y_test.flatten()

    rf = random_forest(X_train_mfcc, y_train, X_val_mfcc, y_val)
    dt = decision_tree(X_train_mfcc, y_train, X_val_mfcc, y_val)
    svm = support_vector_machine(X_train_mfcc, y_train, X_val_mfcc, y_val)
    lr = logistic_regression(X_train_mfcc, y_train, X_val_mfcc, y_val)
    knn = k_nearest_neighbors(X_train_mfcc, y_train, X_val_mfcc, y_val)


    # Create a voting classifier
    estimators = [('rf', rf), ('svm', svm), ('knn', knn)]
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    voting_clf.fit(X_train_mfcc, y_train)
    y_pred_val = voting_clf.predict(X_val_mfcc)
    cf = classification_report(y_val, y_pred_val)
    print(cf)

    y_pred_test = voting_clf.predict(X_test_mfcc)
    cf = classification_report(y_test, y_pred_test)
    print(cf)

    # Plot confusion matrix for test set
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=voting_clf.classes_)

    disp.plot()
    plt.show()

if __name__ == "__main__":
    main()
# %%

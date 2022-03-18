import cv2 #used to load the images (in the future it will have other uses)
import numpy as np
import matplotlib.pyplot as plt #for all ur plotting needs
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


# functii ----------------------------------------------------------------------------------
def show_40_distinct_faces(images, w, h):
   plt.figure(figsize=(13,7))
   plt.subplots_adjust(hspace=.5)
   for i in range(n_classes):
      plt.subplot(4, 10, i + 1)
      plt.imshow(images.iloc[i*10].values.reshape((w, h)), cmap=plt.cm.gray)
      plt.title(f"face ID: {i+1}", size=12)
      plt.xticks(())
      plt.yticks(())
   plt.suptitle("The 40 distinct faces")
   plt.show()


def show_all_faces(images, w, h):
   plt.figure(figsize=(13,7))
   plt.subplots_adjust(hspace=.5)
   for i in range(n_classes):
      plt.subplot(4, 10, i + 1)
      plt.imshow(images.iloc[i].values.reshape((w, h)), cmap=plt.cm.gray)
      plt.title(f"face ID: {int(i/10)}", size=12)
      plt.xticks(())
      plt.yticks(())
   plt.suptitle("The 10 faces of 4 people")
   plt.show()


def plot_eigenfaces(images, w, h):
   plt.figure(figsize=(13,7))
   plt.subplots_adjust(hspace=.5)
   for i in range(n_classes):
      plt.subplot(4, 10, i + 1)
      plt.imshow(images[i].reshape((w, h)), cmap=plt.cm.gray)
      plt.title(f"eigen ID: {i+1}", size=12)
      plt.xticks(())
      plt.yticks(())
   plt.suptitle("Eigen Faces")
   plt.show()


def show_results(images, w, h, y_pred_test):
   plt.figure(figsize=(12,7))
   plt.subplots_adjust(hspace=.5)
   for i in range(18):
      plt.subplot(3, 6, i + 1)
      plt.imshow(images.iloc[i].values.reshape((w, h)), cmap=plt.cm.gray)
      plt.title(f"Predicted: {int(y_pred_test[i])}\nTrue: {int(y_test.iloc[i])}", size=12)
      plt.xticks(())
      plt.yticks(())
   plt.suptitle("Predicted vs. True", size=16)
   plt.show()


def knn_c(n_neighbors, x_train_pca, y_train, x_test_pca, y_test):
   knn_classifier = KNeighborsClassifier(n_neighbors).fit(x_train_pca, y_train)

   y_pred_test = knn_classifier.predict(x_test_pca)
   print(f"Accuracy score knn for k = {n_neighbors}: {accuracy_score(y_test, y_pred_test)*100}%\n")
   print(f"Classification results knn for k = {n_neighbors}:\n{classification_report(y_test, y_pred_test)}\n")
   print(f"Confusion matrix knn for k = {n_neighbors}:\n{confusion_matrix(y_test, y_pred_test, labels=range(n_classes))}\n")

   if(n_neighbors==1):
      accuracy_nn.append(accuracy_score(y_test, y_pred_test)*100)
   if(n_neighbors==3):
      accuracy_3nn.append(accuracy_score(y_test, y_pred_test) * 100)

   show_results(x_test, w, h, y_pred_test)
   print(f"Prediction knn for k = {n_neighbors}:")
   data = {
      'True test images': np.array(y_test).astype(int),
      'Predicted test images': y_pred_test.astype(int),
      'Status': ["true" if np.array(y_test)[i] == y_pred_test[i] else "false" for i in range(len(y_test))],
   }
   result = pd.DataFrame(data=data)
   print(f"{result}\n")
   print(f"Number of predicted classes knn for k = {n_neighbors}: {data['Status'].count('true')}/{len(data['Status'])}\n")


def kmeans_c(n_clusters, x_train_pca, y_train, x_test_pca, y_test):
   kmeans = KMeans(n_clusters).fit(x_train_pca)
   reference_labels = retrieve_info(kmeans.labels_, y_train, kmeans)
   number_labels = np.random.rand(len(kmeans.labels_))
   for i in range(len(kmeans.labels_)):
      number_labels[i] = reference_labels[kmeans.labels_[i]]

   y_pred_test_2 = []
   for i in range(x_test_pca.shape[0]):
      img = x_test_pca[i].reshape(1, x_test_pca[i].shape[0])
      predicted_cluster = kmeans.predict(img)
      y_pred_test_2.append(reference_labels[predicted_cluster[0]])

   print(f"Accuracy score kmeans: {accuracy_score(y_test, y_pred_test_2)*100}%\n")
   print(f"Classification results kmeans:\n{classification_report(y_test, y_pred_test_2)}\n")
   print(f"Confusion matrix kmeans:\n{confusion_matrix(y_test, y_pred_test_2, labels=range(n_classes))}\n")

   accuracy_kmeans.append(accuracy_score(y_test, y_pred_test_2)*100)

   show_results(x_test, w, h, y_pred_test_2)
   data = {
      'True test images': np.array(y_test).astype(int),
      'Predicted test images': y_pred_test_2,
      'Status': ["true" if np.array(y_test)[i] == y_pred_test_2[i] else "false" for i in range(len(y_test))],
   }
   result = pd.DataFrame(data=data)
   print(f"{result}\n")
   print(f"Number of predicted classes kmeans: {data['Status'].count('true')}/{len(data['Status'])}\n")


def retrieve_info(cluster_labels, y_train, kmeans):
   reference_labels = {}
   for i in range(len(np.unique(kmeans.labels_))):
      index = np.where(cluster_labels == i,1,0)
      count = np.bincount(y_train[index==1])
      num = count.argmax()
      count = list(count)
      count = count[:40]
      count.extend(np.zeros(40 - len(count)).astype(int))
      # if(i<6):
      #  plt.bar(range(len(count)), count)  # amount of variance that is captured by each component, cumulative sum
      #  plt.title(f"Cluster: {i}, target: {num}", size=12)
      #  plt.grid()
      #  plt.show()
      reference_labels[i] = num
   # print(reference_labels)
   return reference_labels


# import -----------------------------------------------------------------------------------
# import files
dataBase = "D:\\Folder Alexandra\\Scoala\\FACULTATE\\MASTER ANUL I\\IC1\\Proiect\\Baza de date\\" #import database
files = []
for i in range(1, 401):
   files.append(dataBase + str(i) + ".bmp") #write the paths in files

# get images from files
faces_original = [] # vector to store images
for file in files:
   faces_original.append(cv2.imread(file)) #get images form paths
faces = np.array(faces_original)
faces_g = []
for file in faces_original:
   faces_g.append(cv2.cvtColor(file, cv2.COLOR_RGB2GRAY)) #make images to have one color channel
faces_g = np.array(faces_g)
print(f'Initial set dimensions: {faces_g.shape}')
w = faces_g.shape[1]
h = faces_g.shape[2]
n_classes = 40 #number of classes
print(f"There are {len(faces_g)} images in the dataset")
print(f"There are {n_classes} unique targets in the dataset")
print(f"Size of each image is {h} x {w}\n")

# reduce dimensions
faces = pd.DataFrame([]) #make faces dataframe
i = 0
name = 0
for face_g in faces_g:
   face = pd.Series(face_g.flatten(), name = i)
   faces = faces.append(face)
   i += 1
print(f'Set dimensions after reducing: {faces.shape}\n')

# add the target column
face_tg = [(str(index) + " ") * 10 for index in range(1,41)]
face_tg = np.array([index.split(" ")[:-1] for index in face_tg]).astype(int)
faces_target = []
for i in face_tg:
   for j in i:
      faces_target.append(j)
targets = np.reshape(face_tg, (len(faces_target), 1))
faces_all = np.concatenate([faces, targets], axis=1)
face_dataset= pd.DataFrame(faces_all)
features = range(10304)
features_labels = np.append(features, 'target')
face_dataset.columns = features_labels
print(f"Table with pixel values :\n{face_dataset}\n") #show table

# define pixels and labels
labels = face_dataset["target"]
pixels = face_dataset.drop(["target"], axis=1)
show_40_distinct_faces(pixels, w, h)
show_all_faces(pixels, w, h)


# set parameters for test ------------------------------------------------------------------
test_size = 0.1
n_components_vect = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360]
accuracy_nn = []
accuracy_3nn = []
accuracy_kmeans = []
for i in n_components_vect:
   print(f"n_components = {i}")
   n_components = i

   # split dataset in train and test -------------------------------------------------------
   x_train, x_test, y_train, y_test = train_test_split(pixels, labels, test_size = test_size, random_state = 42) # x-features, y-targets
   print(f"Number of train images: {x_train.shape[0]}, which is {x_train.shape[0]*100/400}% from the total number of images")
   print(f"Number of test images: {x_test.shape[0]}, which is {x_test.shape[0]*100/400}% from the total number of images\n")


   # perform PCA ---------------------------------------------------------------------------
   pca = PCA(n_components).fit(x_train)
   eigenfaces = pca.components_.reshape((n_components, h, w)) #find the eigenfaces
   # plot_eigenfaces(eigenfaces, w, h)


   # project training data to PCA ----------------------------------------------------------
   x_train_pca = pca.transform(x_train)
   x_test_pca = pca.transform(x_test)
   print(f"Train set dimension before PCA: {x_train.shape}")
   print(f"Train set dimension before PCA: {x_train_pca.shape}")
   print(f"Test set dimension before PCA: {x_test.shape}")
   print(f"Test set dimension before PCA: {x_test_pca.shape}\n")


   # knn -----------------------------------------------------------------------------------
   print(f"NN CLASIFIER")
   knn_c(1, x_train_pca, y_train, x_test_pca, y_test)
   print(f"3-NN CLASIFIER")
   knn_c(3, x_train_pca, y_train, x_test_pca, y_test)


   # K-means -------------------------------------------------------------------------------
   print(f"K-MEANS CLASIFIER")
   kmeans_c(40, x_train_pca, y_train, x_test_pca, y_test)

plt.plot(n_components_vect,accuracy_nn)
plt.title(f"Acuratetea pentru NN - {x_train.shape[0]} imagini de antrenare", size=12)
plt.ylim([0, 100])
plt.grid()
plt.show()

plt.plot(n_components_vect,accuracy_3nn)
plt.title(f"Acuratetea pentru 3NN - {x_train.shape[0]} imagini de antrenare", size=12)
plt.ylim([0, 100])
plt.grid()
plt.show()

plt.plot(n_components_vect,accuracy_kmeans)
plt.title(f"Acuratetea pentru k-means - {x_train.shape[0]} imagini de antrenare", size=12)
plt.ylim([0, 100])
plt.grid()
plt.show()

# afisarea acuratetii in functie de dimensiunea setului de antrenare -----------------------
ox = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
oyNN = [51.55, 68.43, 81.78, 81.25, 91.50, 94.37, 99.16, 98.75, 100.0]
oy3NN = [28.05, 46.87, 64.28, 67.50, 82.50, 85.62, 89.16, 95.00, 94.50]
oyKmeans = [51.11, 52.50, 58.57, 61.25, 70.50, 66.25, 70.83, 77.50, 80.00]

plt.plot(ox,oyNN)
plt.title(f"Acuratetea pentru NN in functie de dimensiunea setului de antrenare", size=12)
plt.ylim([0, 100])
plt.xlabel('Procent din setul de antrenare')
plt.ylabel('Acuratetea')
plt.grid()
plt.show()

plt.plot(ox,oy3NN)
plt.title(f"Acuratetea pentru 3NN in functie de dimensiunea setului de antrenare", size=12)
plt.ylim([0, 100])
plt.xlabel('Procent din setul de antrenare')
plt.ylabel('Acuratetea')
plt.grid()
plt.show()

plt.plot(ox,oyKmeans)
plt.title(f"Acuratetea pentru k-means in functie de dimensiunea setului de antrenare", size=12)
plt.ylim([0, 100])
plt.xlabel('Procent din setul de antrenare')
plt.ylabel('Acuratetea')
plt.grid()
plt.show()

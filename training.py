import numpy as np
import cv2
import os
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# WSTĘPNE PRZETWARZANIE OBRAZU #

def grayscale(img):  # Funkcja grayscale(img) przekształca obraz kolorowy w obraz w skali szarości za pomocą funkcji cv2.cvtColor() z biblioteki OpenCV.

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):  # Funkcja equalize(img) wykorzystuje funkcję cv2.equalizeHist() do wyrównania histogramu obrazu, co powoduje poprawę kontrastu i poprawę jakości obrazu.
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img): # Funkcja preprocessing(img) stosuje obie powyższe funkcje do przetworzenia obrazu. Następnie normalizuje wartości pikseli obrazu, dzieląc je przez 255, aby uzyskać wartości z zakresu od 0 do 1, co ułatwi uczenie sieci neuronowej.
    img = grayscale(img)     # KONWERTOWANIE DO SKALI SZAROŚCI
    img = equalize(img)      # POPRAWA KONTRASTU
    img = img/255            # NORMALIZACJA WARTOŚCI PIKSELA
    return img

# PODSTAWOWE PARAMETRY #
 
path = "DaneBadawcze"  # folder z danymi treningowymi
labelFile = 'EtykietyDanych.csv'  # plik z nazwami etykiet
batch_size_value: int = 50  # przechowuje liczbę obrazów, które będą przetwarzane jednocześnie w trakcie uczenia sieci neuronowej.
steps_per_epoch_value = 446  # przechowuje liczbę kroków (batch'y) wykonywanych w każdej epoce treningowej.
epoch_value = 100  # przechowuje liczbę epok, czyli liczbę przejść przez cały zbiór treningowy, które zostaną wykonane podczas uczenia sieci.
imageDimensions = (32, 32, 3)  # przechowuje wymiary obrazów w formacie (szerokość, wysokość, liczba kanałów).
testRatio = 0.2  # przechowuje procentowy udział obrazów przeznaczonych do testowania w stosunku do całego zbioru treningowego. W tym przypadku wynosi on 30%.
validationRatio = 0.2  # przechowuje procentowy udział obrazów przeznaczonych do walidacji w stosunku do zbioru treningowego (po odjęciu obrazów testowych). W tym przypadku wynosi on również 30%.
 
# IMPORTOWANIE OBRAZÓW #

licznik = 0  # zmienna licznik inicjalizuje się na 0 i służy do numerowania klas
listaObrazow = []  # zmienna lisatObrazow inicjalizuje pustą listę, w której będą przechowywane obrazy.
etykietyClass = []  # zmienna etykietyClass inicjalizuje pustą listę, w której będą przechowywane etykiety klas.
listaPlikow = os.listdir(path)  # funkcja os.listdir() pobiera listę plików i folderów w podanym folderze path i zapisuje ją do zmiennej listaPlikow.
print("Łączna liczba wykrytych zbiorów danych:", len(listaPlikow))  # wyświetla liczbę wykrytych klas w folderze.
liczbaClass = len(listaPlikow)  # zmienna liczbaClass przechowuje liczbę klas.
print("Importowanie danych ...")  # wyświetla komunikat informujący o rozpoczęciu importowania danych.
for i in range(0, len(listaPlikow)):  # pętla for przechodzi przez każdą klasę w folderze.
    listaClass = os.listdir(path+"/"+str(licznik))  # funkcja os.listdir() pobiera listę plików w podfolderze count w folderze path i zapisuje ją do zmiennej myPicList.
    for j in listaClass:  # pętla for przechodzi przez każdy plik w podfolderze count.
        obecnyObraz = cv2.imread(path+"/"+str(licznik)+"/"+j)  # funkcja cv2.imread() wczytuje obraz z pliku o ścieżce path+"/"+str(count)+"/"+y i zapisuje go do zmiennej curImg.
        listaObrazow.append(obecnyObraz)  # dodaje obraz do listy listaObrazow.
        etykietyClass.append(licznik)  # dodaje etykietę klasy do listy etykietyClass.
    print(licznik, end=" ")  # wyświetla numer aktualnie przetwarzanej klasy.
    licznik += 1  # zwiększa wartość licznik o 1, aby przejść do przetwarzania kolejnej klasy.
print(" ")
listaObrazow = np.array(listaObrazow)  # zamienia listę listaObrazow na tablicę NumPy.
etykietyClass = np.array(etykietyClass)  # zamienia listę etykietyClass na tablicę NumPy.

# PODZIAŁ DANYCH  NA ZBIORY TRENENINGOWE, TERSTOWE i WALIDACYJNE #

X_train, X_test, y_train, y_test = train_test_split(listaObrazow, etykietyClass, test_size=testRatio)  # funkcja train_test_split() losowo dzieli zbiór images i classNo na zbiory treningowy X_train i y_train oraz testowy X_test i y_test, przy zachowaniu proporcji określonych przez testRatio.
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)  # funkcja train_test_split() losowo dzieli zbiór X_train i y_train na zbiory treningowy X_train i y_train oraz walidacyjny X_validation i y_validation, przy zachowaniu proporcji określonych przez validationRatio.
 
# X_train = zawiera tablicę obrazów, które będą używane do treningu modelu.
# y_train = zawiera odpowiadające numery klas dla obrazów w X_train.
# x_test zawiera tablicę obrazów, które będą używane do testowania modelu.
# y_test zawiera odpowiadające numery klas dla obrazów w X_test.

# SPRAWDZENIE  POPRAWNOŚCI PODZIAŁU DANYCH NA ZBIÓR TRENINGOWY, WALIDACYJNY I TESTOWY

print("Kształt Danych")
print(" Trening", end="")
print(X_train.shape, y_train.shape)
print(" Validacja", end="")
print(X_validation.shape, y_validation.shape)
print(" Test", end="")
print(X_test.shape, y_test.shape)
assert(X_train.shape[0] == y_train.shape[0]), "Liczba obrazów nie jest równa liczbie etykiet w zbiorze treningowym"
assert(X_validation.shape[0] == y_validation.shape[0]), "Liczba obrazów nie jest równa liczbie etykiet w zbiorze validacji"
assert(X_test.shape[0] == y_test.shape[0]), "Liczba obrazów nie jest równa liczbie etykiet w zestawie testowymt"
assert(X_train.shape[1:] == imageDimensions), " Wymiary obrazów szkoleniowych są nieprawidłowe "
assert(X_validation.shape[1:] == imageDimensions), " Wymiary obrazów validacji są nieprawidłowe "
assert(X_test.shape[1:] == imageDimensions), " Wymiary obrazów testowych są nieprawidłowe"

# WCZYTAJ PLIK CSV

data = pd.read_csv(labelFile)  # Ten kod wczytuje plik CSV zawierający etykiety klas i wyświetla jego kształt oraz typ danych
print("Kształt danych CSV ", data.shape, type(data))



# ITERACJA I PRZETWARZANIE #

X_train = np.array(list(map(preprocessing, X_train)))  #iteruje przez każdy obraz w X_train, przetwarza każdy obraz zgodnie z funkcją preprocessing i zwraca listę przetworzonych obrazów a następnie przypisuje przetworzone obrazy do X_train i przekształca je na numpy array.
X_validation = np.array(list(map(preprocessing, X_validation))) #iteruje przez każdy obraz w X_validation, przetwarza każdy obraz zgodnie z funkcją preprocessing i zwraca listę przetworzonych obrazów a następnie przypisuje przetworzone obrazy do X_train i przekształca je na numpy array.
X_test=np.array(list(map(preprocessing, X_test))) #iteruje przez każdy obraz w x_test, przetwarza każdy obraz zgodnie z funkcją preprocessing i zwraca listę przetworzonych obrazów a następnie przypisuje przetworzone obrazy do X_train i przekształca je na numpy array.
# cv2.imshow("GrayScale Images",X_train[random.randint(0,len(X_train)-1)]) # losowo wybiera obraz ze zbioru szkoleniowego X_train i wyświetla go jako obraz w skali szarości w oknie o nazwie "GrayScale Images". Służy do sprawdzenia, czy przetwarzanie obrazów zostało wykonane poprawnie
 
# DODANIE DODATKOWEGO WYMIARU #

X_train=X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1) # Dodaje jedny dodatkowy wymiar do każdego zbioru obrazów - wymiar głębokości, który jest ustawiany na 1. W ten sposób, każdy obraz jest reprezentowany jako macierz o wymiarach (wysokość, szerokość, 1), co jest wymagane przez model sieci neuronowych w bibliotece Keras. Funkcja reshape jest używana do zmiany kształtu macierzy obrazów.
X_validation=X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test=X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
 
# AUGUMENTACJA, GENEROWANIE DODATKOWYCH  OBRAZÓW #

dataGen= ImageDataGenerator( #ImageDataGenerator to klasa z biblioteki keras.preprocessing.image, która pozwala na łatwe generowanie nowych obrazów na podstawie istniejących danych treningowych.
                            width_shift_range=0.1,   # Przesunięcie poziome 0.1 = 10%
                            height_shift_range=0.1,  # Przesunięcie pionowe
                            zoom_range=0.2,  # Powiększenie 0.2 Oznacza że może przejść oc 0.8 do 1.2
                            shear_range=0.1,  # Skręcenie
                            rotation_range=10)  # Obrót
dataGen.fit(X_train)  #  Generator jest dostosowywany do danych treningowych, aby określić odpowiednie parametry augmentacji.
batches= dataGen.flow(X_train, y_train, batch_size=20)  # Generuje losowe batche (porcje) danych treningowych i ich etykiet w czasie rzeczywistym. Tutaj określono wielkość porcji danych batch_size=20.
X_batch, y_batch = next(batches)  # zwraca następny batch danych treningowych i ich etykiet.

 
y_train = to_categorical(y_train, liczbaClass)  # y_train, y_validation i y_test są kodowane przy użyciu to_categorical z biblioteki keras.utils, aby przekształcić etykiety z postaci numerycznej na wektor jednostkowy (one-hot encoding), co jest wymagane przez model sieci neuronowej.
y_validation = to_categorical(y_validation, liczbaClass)
y_test = to_categorical(y_test, liczbaClass)
 
# SPLOTOWY MODEL SIECI NEURONOWEJ #
def modelUczenia():
    no_Of_Filters = 100  # liczba filtrów konwolucyjnych użytych w pierwszej warstwie konwolucyjnej.
    size_of_Filter = (5, 5)  # rozmiar jądra filtru konwolucyjnego. Przesuwa się po obrazie, aby uzyskać cechy.
    size_of_Filter2 = (3, 3)  # rozmiar jądra filtru konwolucyjnego w drugiej warstwie konwolucyjnej.
    size_of_pool = (2, 2)  # rozmiar okna przesuwającego się po obrazie w warstwie pooling. W celu zwiększenia ogólności modelu i zmniejszenia overfittingu.
    no_Of_Nodes = 2000   # liczba neuronów w warstwie ukrytej.
    model = Sequential()  # definicja modelu sieci neuronowej, tworzenie pustego modelu.
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu')))  # - dodanie warstwy konwolucyjnej do modelu. input_shape to wymiary obrazu wejściowego.
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu'))) # dodanie kolejnej warstwy konwolucyjnej do modelu.
    model.add(MaxPooling2D(pool_size=size_of_pool))  # dodanie warstwy pooling do modelu.
 
    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2, activation='relu')))  # dodanie kolejnej warstwy konwolucyjnej z mniejszą liczbą filtrów do modelu.
    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2, activation='relu')))  # dodanie jeszcze jednej warstwy konwolucyjnej z mniejszą liczbą filtrów do modelu.
    model.add(MaxPooling2D(pool_size=size_of_pool))  # dodanie kolejnej warstwy pooling do modelu.
    model.add(Dropout(0.5))  # dodanie warstwy dropout do modelu, która losowo usuwa węzły podczas uczenia, aby zapobiec overfittingowi.
 
    model.add(Flatten())  # spłaszczanie warstw konwolucyjnych w jednowymiarowy wektor.
    model.add(Dense(no_Of_Nodes, activation='relu'))  # dodanie warstwy w pełni połączonej z określoną liczbą neuronów i funkcją aktywacji ReLU.
    model.add(Dropout(0.5))  # dodanie kolejnej warstwy dropout do modelu, aby zapobiec overfittingowi.
    model.add(Dense(liczbaClass, activation='softmax'))  # dodanie warstwy wyjściowej z określoną liczbą klas i funkcją aktywacji softmax.

    # KOMPILACJA MODELU #

    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])  # funkcja compile ustawia parametry uczenia modelu, w tym algorytm optymalizacji (w tym przypadku Adam), szybkość uczenia (learning rate), funkcję straty (loss ) oraz metryki, które zostaną użyte do oceny modelu podczas uczenia (tutaj 'accuracy' oznacza dokładność).
    return model

 
# TRENING MODELU

model = modelUczenia()  # tworzy instancję klasy myModel() i przypisuje ją do zmiennej model
print(model.summary())  # wyświetla podsumowanie architektury modelu za pomocą metody summary().
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_value), steps_per_epoch=steps_per_epoch_value, epochs=epoch_value, validation_data=(X_validation, y_validation), shuffle=1)
# Model jest trenowany na danych treningowych i walidacyjnych za pomocą metody fit().
# Dane treningowe i walidacyjne są pobierane za pomocą funkcji flow() z obiektu dataGen - generatora danych obrazowych.
# batch_size_val określa liczbę obrazów, które są przetwarzane jednocześnie w trakcie jednej iteracji uczenia
# steps_per_epoch_val określa liczbę kroków (batch'y) wykonywanych w każdej epoce treningowej
# epochs_val to liczba epok, czyli liczba przejść przez cały zbiór treningowy, które zostaną wykonane podczas uczenia sieci.
# validation_data to dane walidacyjne, a shuffle ustawione na 1 oznacza, że dane treningowe będą przetasowywane przed każdą epoką treningową.
 
# ZAPISANIE MODELU JAKO PLIKU HDF5

model.save('my_model.h5')   # tworzy plik HDF5 o nazwie 'my_model.h5', w którym zapisywany jest wytrenowany model.
model1 = load_model('my_model.h5')  # Wczytuje model z pliku 'my_model.h5' i przypisuje go do zmiennej 'model1'.
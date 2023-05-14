
import numpy as np # import biblioteki numpy i nadanie jej nazwy "np"
import cv2 # import biblioteki OpenCV
from keras.models import load_model

#############################################

frameWidth= 1280 # szerokość kamery
frameHeight = 720 # wysokość kamery
brightness = 160 # jasność obrazu
threshold = 0.85 # próg prawdopodobieństwa
font = cv2.FONT_HERSHEY_SIMPLEX # wybór fontu
##############################################
 
# KONFIGURACJA KAMERY #

cap = cv2.VideoCapture(0)  # tworzony jest obiekt klasy cv2.VideoCapture, który umożliwia przechwytywanie strumienia wideo z kamery.
cap.set(3, frameWidth)  # ustawia szerokość ramki na wartość frameWidth.
cap.set(4, frameHeight)  # ustawia wysokość ramki na wartość frameHeight
cap.set(10, brightness)  # ustawia jasność obrazu na wartość brightness.

# IMPORT WYTRENOWANEGO MODELU #

model= load_model('my_model.h5')  # załadowanie wytrenowanego modelu
 
def grayscale(img): # Funkcja grayscale(img) przekształca obraz kolorowy w obraz w skali szarości za pomocą funkcji cv2.cvtColor() z biblioteki OpenCV.
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):  # Funkcja equalize(img) wykorzystuje funkcję cv2.equalizeHist() do wyrównania histogramu obrazu, co powoduje poprawę kontrastu i poprawę jakości obrazu.
    img =cv2.equalizeHist(img)
    return img

def preprocessing(img):  # Funkcja preprocessing(img) stosuje obie powyższe funkcje do przetworzenia obrazu. Następnie normalizuje wartości pikseli obrazu, dzieląc je przez 255, aby uzyskać wartości z zakresu od 0 do 1, co ułatwi uczenie sieci neuronowej.
    img = grayscale(img)     # KONWERTOWANIE DO SKALI SZAROŚCI
    img = equalize(img)      # POPRAWA KONTRASTU
    img = img/255            # NORMALIZACJA WARTOŚCI PIKSELA
    return img

def getClassName(classNo):  #  Zwraca nazwę klasy na podstawie przypisanego numeru klasy
    if (classNo == 0).any(): return 'Ograniczenie predkosci 20 km/h'
    elif (classNo == 1).any(): return 'Ograniczenie predkosci 30 km/h'
    elif (classNo == 2).any(): return 'Ograniczenie predkosci 50 km/h'
    elif (classNo == 3).any(): return 'Ograniczenie predkosci 60 km/h'
    elif (classNo == 4).any(): return 'Ograniczenie predkosci 70 km/h'
    elif (classNo == 5).any(): return 'Ograniczenie predkosci 80 km/h'
    elif (classNo == 6).any(): return 'Koniec ograniczenia predkosci 80 km/h'
    elif (classNo == 7).any(): return 'Ograniczenie prędkosci 100 km/h'
    elif (classNo == 8).any(): return 'Ograniczenie prędkosci 120 km/h'
    elif (classNo == 9).any(): return 'Zakaz wyprzedzania'
    elif (classNo == 10).any(): return 'Zakaz wyprzedzania cięzarowek'
    elif (classNo == 11).any(): return 'Pierwszenstwo przejazdu'
    elif (classNo == 12).any(): return 'Droga z pierwszenstwem'
    elif (classNo == 13).any(): return 'Ustap pierwszenstwa'
    elif (classNo == 14).any(): return 'Stop'
    elif (classNo == 15).any(): return 'Zakaz wjazdu'
    elif (classNo == 16).any(): return 'Zakaz wjazdu dla cięzarowek'
    elif (classNo == 17).any(): return 'Zakaz wjazdu (ogolny)'
    elif (classNo == 18).any(): return 'Uwaga'
    elif (classNo == 19).any(): return 'Niebezpieczny zakret w lewo'
    elif (classNo == 20).any(): return 'Niebezpieczny zakret w prawo'
    elif (classNo == 21).any(): return 'Podwojna krzywa'
    elif (classNo == 22).any(): return 'Nierowna droga'
    elif (classNo == 23).any(): return 'Sliska droga'
    elif (classNo == 24).any(): return 'Droga zweza się z prawej strony'
    elif (classNo == 25).any(): return 'Roboty drogowe'
    elif (classNo == 26).any(): return 'Sygnalizacja swietlna'
    elif (classNo == 27).any(): return 'Przejscie dla pieszych'
    elif (classNo == 28).any(): return 'Przejscie dla dzieci'
    elif (classNo == 29).any(): return 'Przejscie dla rowerow'
    elif (classNo == 30).any(): return 'Uwaga na gołoledz lub snieg'
    elif (classNo == 31).any(): return 'Zwierzeta na drodze'
    elif (classNo == 32).any(): return 'Koniec ograniczenia predkosci i zakazu wyprzedzania'
    elif (classNo == 33).any(): return 'Skret w prawo'
    elif (classNo == 34).any(): return 'Skret w lewo'
    elif (classNo == 35).any(): return 'Tylko prosto'
    elif (classNo == 36).any(): return 'Prosto lub w prawo'
    elif (classNo == 37).any(): return 'Prosto lub w lewo'
    elif (classNo == 38).any(): return 'Prawostronne rondo'
    elif (classNo == 39).any(): return 'Lewostronne rondo'
    elif (classNo == 40).any(): return 'Obowiazkowe rondo'
    elif (classNo == 41).any(): return 'Koniec zakazu wyprzedzania'
    elif (classNo == 42).any(): return 'Koniec zakazu wyprzedzania dla pojazdow o masie powyzej 3,5 tony'


while True:
 
# WCZYTANIE OBRAZU Z KAMERY #

    success, obrazKamera = cap.read()  #  odczytuje klatkę wideo z kamery i zwraca wartość True, jeśli klatka została odczytana poprawnie, a False, jeśli nie udało się odczytać klatki. Klatka jest zapisywana w zmiennej obrazKamery jako obraz w formacie numpy array.
 
# PRZETWARZANIE OBRAZU Z KAMERY #

    img = np.asarray(obrazKamera)  #  zamienia obraz z kamery na tablicę numpy, aby mógł być przetwarzany przez bibliotekę OpenCV.
    img = cv2.resize(img, (32, 32))  #  zmienia rozmiar obrazu na 32x32 pikseli, co jest wymaganym rozmiarem wejściowym dla sieci neuronowej
    img = preprocessing(img)  # stosuje operacje preprocessingu na obrazie, takie jak normalizacja, konwersja na odcienie szarości, wykrycie krawędzi itp., aby ułatwić rozpoznawanie znaku przez sieć neuronową.
    img = img.reshape(1, 32, 32, 1)  # zmienia kształt tablicy numpy na wymagany przez sieć neuronową kształt wejściowy (1,32,32,1), gdzie 1 oznacza liczbę przykładów, 32x32 to rozmiar obrazu, a 1 oznacza liczbę kanałów (w tym przypadku 1 kanał dla obrazów w skali szarości).
    cv2.putText(obrazKamera, "ROZPOZNANY ZNAK: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)  # umieszcza napis "ROZPOZNANY ZNAK:" na obrazie, aby wskazać miejsce, gdzie zostanie wyświetlony wynik rozpoznawania.
    cv2.putText(obrazKamera, "PRAWDOPODOBIENSTWO: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)  # - umieszcza napis "PRAWDOPODOBIENSTWO:" na obrazie, aby wskazać miejsce, gdzie zostanie wyświetlony wynik prawdopodobieństwa dla rozpoznanego znaku.

    #    Przewidywanie obrazu
    predictions = model.predict(img)  # Metoda predict wykorzystuje wcześniej wytrenowany model, aby dokonać predykcji na przesłanym obrazie img.
    classIndex = np.argmax(predictions)  #  Zmienna classIndex przechowuje indeks klasy, którą model przewidział jako najlepszą.
    probabilityValue = np.amax(predictions)  #  Zmienna probabilityValue przechowuje prawdopodobieństwo dla tej klasy.



    if probabilityValue > threshold:  # Jeśli wartość prawdopodobieństwa przekracza próg (threshold), to na obrazie kamery zostaje wyświetlona informacja o rozpoznanym znaku oraz prawdopodobieństwie jego poprawnego rozpoznania.
        cv2.putText(obrazKamera, str(classIndex) + " " + str(getClassName(classIndex)).encode('utf-8').decode('utf-8'), (320, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(obrazKamera, str(round(probabilityValue*100,2)) + "%", (330, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Program do rozpoznawania znakow drogowych", obrazKamera) # wyświetla obraz z kamerki z naniesionymi informacjami o rozpoznanym znaku i jego prawdopodobieństwie na wystąpienie. Wykorzystywana jest funkcja imshow() z biblioteki OpenCV, która wyświetla obraz w nowym oknie o podanej nazwie.

    if cv2.waitKey(1) == 13:  # Jeśli zostanie naciśnięty klawisz Enter (13), pętla przerywa działanie i zamykane są wszystkie okna.
        break
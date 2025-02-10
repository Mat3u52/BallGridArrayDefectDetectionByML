# Wykrywanie wad Ball Grid Array za pomocą ML

### Rozpoznanie defektu typu void dla układu "Ball Grid Array" przy pomocy machine learning.

IPC-A-610, zatytułowany „Akceptowalność zespołów elektronicznych”, jest powszechnie uznawanym standardem w branży elektronicznej, który zapewnia kompleksowe kryteria akceptacji zespołów elektronicznych, w tym tych wykorzystujących komponenty Ball Grid Array (BGA).

Jeśli chodzi o BGA, IPC-A-610 zajmuje się kilkoma krytycznymi aspektami, aby zapewnić jakość montażu:

Jakość połączenia lutowanego: Norma określa akceptowalne warunki dla połączeń lutowanych BGA, skupiając się na takich czynnikach, jak przesunięcie kulki lutowniczej, kształt i obecność pustych przestrzeni. Na przykład IPC-A-610F definiuje, że kulka BGA jest akceptowalna dla klas 1, 2 i 3, jeśli jej powierzchnia pusta jest mniejsza niż 30% powierzchni kuli w 2D kontroli rentgenowskiej.

Inspekcja i testowanie: Norma określa zalecane techniki inspekcji, takie jak analiza rentgenowska, w celu oceny integralności połączeń lutowanych BGA i wykrywania potencjalnych wad, takich jak pustki lub nieprawidłowe ustawienie.

Przestrzeganie kryteriów IPC-A-610 dla zespołów BGA pomaga zapewnić niezawodność i wydajność produktu, co jest szczególnie ważne w zastosowaniach wymagających wysokiej jakości zespołów elektronicznych.

PCB:
<img src="https://static.wixstatic.com/media/74cd3b_773e6cd81fb1433e972e27dbf33c53bd~mv2_d_4000_3000_s_4_2.jpg" alt="PCB" width="200">
- Źródło: [regulus-ems.com](https://www.regulus-ems.com/leaded-pcb-assembly)

BGA:
<img src="https://www.fs-pcba.com/wp-content/uploads/2022/12/1-9.jpg" alt="BGA" width="200">
- Źródło: [madpcb.com](https://madpcb.com/glossary/pbga/)


BGA solder ball:
<img src="https://epp-europe-news.com/wp-content/uploads/3/6/3603682.jpg" alt="Ball" width="200">

Źródło: [epp-europe-news.com](https://epp-europe-news.com/technology/applications/proper-inspection-strategy/)


X-Ray image:

| Przykład dobrych połączeń                                                                                                                      | Przykład wadliwych połączeń                                                                                                                      |
|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/good_image.jpg?raw=true" alt="Good BGA" width="200">  | <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/not_good_image.jpg?raw=true" alt="Bad BGA" width="200"> |
| <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/single_good_ball.png?raw=true" alt="Good BGA" width="200">  | <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/single_not_good_ball.png?raw=true" alt="Good BGA" width="200"> |



## 1. Przygotowanie zestawu danych
- Informacje na temat przygotowanego wcześniej pliku "Dataset.zip" Przygotowana paczka zawiera zestaw różnych formatów plików w tym zdjęci ATG układu BGA po montażu w formacie .png
  - Metoda "print_zip_summary":
  
          Podsumowanie zawartości pliku ZIP.
          Po podaniu ścieżki do pliku ZIP ta funkcja otwiera plik w trybie tylko do odczytu,
          pobiera listę wszystkich plików w nim zawartych i wyświetla podsumowanie pliku ZIP.
          
          W szczególności wyświetla:
            Nazwę bazową pliku ZIP. 
            Rozmiar pliku archiwum ZIP w bajtach.
            Całkowitą liczbę plików zawartych w archiwum.
            Listę ostatnich 5 plików w archiwum. Jeśli archiwum ZIP zawiera mniej niż 5 plików, wydrukuje je wszystkie.
    <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/zipInfo.png?raw=true" alt="zipInfo" width="400" > 
  
  - Metoda "unzip_precess":
        
        Wypakowuje pliki z archiwum ZIP do określonego katalogu, pokazując postęp wypakowywania.
  
  - Metoda "tree":
  
        Rekurencyjnie wyświetla zawartość katalogu w ustrukturyzowanym, hierarchicznym formacie.
    <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/treeInfo.png?raw=true" alt="tree" width="400" >
  
  - Metoda "show_samples":
  
        Losowo wybiera i wyświetla pięć obrazów .png z danego katalogu. Każdy obraz jest zmieniany na
        300x300 pikseli i wyświetlany w oknie pop-up przez 5 sekund. Na ten moment zestaw danych jest bardzo "zanieczyszczony" i nadmiarowy.
    <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/showSamples.png?raw=true" alt="showSamples" width="400" >
  
  - Metoda "recognizer":

        Przetwarza obraz w celu zidentyfikowania największego okrągłego obiektu, 
        oblicza jego średnicę i powierzchnię oraz podświetla puste obszary w obrębie marginesu wokół wykrytego konturu.
        Funkcja zostaje wywołana w pętli 'for' w celu odseparowanie zdjęć z defektem i bez defektu. 
        Dodatkowo w katalogu 'bitmapDiagnostics' umieszczone zostają pliki diagnostyczne.
    <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/diagnostic.png?raw=true" alt="diagnostic" width="400" >

    <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/passFail.png?raw=true" alt="passFail" width="400" >

## 2. Analiza danych (conda jupyter notebook)
  Podczas procesu „recognizer” został wygenerowany plik "SolderBallsSize.csv", zawierający następujące dane:
  - BallDiameter [px] – średnica kulki w pikselach
  - BallArea [px] – powierzchnia kulki w pikselach
  - VoidArea [px] – powierzchnia pustki w pikselach
  - Status [bool] – status (wartość logiczna)

  <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/df.png?raw=true" alt="Good BGA" width="400">

  Wydzielamy pierwsze 1000 etykiet klasy odpowiadających statusom True oraz False i przekształcamy je w dwie kategorie symbolizowane liczbami całkowitymi: 1 (True) i -1 (False), które przypisujemy do wektora y.

  Następnie, ze zbioru 1000 przykładów uczących wydzielamy drugą kolumnę cech (BallArea) oraz trzecią kolumnę (VoidArea), a uzyskane wartości przypisujemy do macierzy cech X. Tak skonstruowane dane możemy zwizualizować jako dwuwymiarowy wykres punktowy.

  <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/scatterPlot.png?raw=true" alt="scatterPlot" width="400">

  - Metoda "perceptron":
    Pierwotna reguła uczenia perceptronu, opracowana przez Franka Rosenblatta, przedstawia się następująco i można ją opisać w kilku etapach:

    1. Ustaw wagi na 0 lub niewielkie, losowe wartości.
    2. Dla każdego przykładu uczącego 𝑥:
         - Oblicz wartość wyjściową y.
         - Zaktualizuj wagi.

    <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/perceptron.jpg?raw=true" alt="perceptron" width="800" >
    
        ppn = Perceptron(eta=0.1, n_iter=10)
        ppn.fit(X,y)
        plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
        plt.xlabel('Epoki')
        plt.ylabel('Liczba aktualizacji')
        plt.show()
    <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/perceptronPlot.png?raw=true" alt="perceptron" width="800" >
  
  Zmniejszenie liczby aktualizacji: Na początku liczba aktualizacji jest wysoka (ponad 30), ale szybko maleje w kolejnych epokach. Oznacza to, że model stopniowo uczy się lepiej klasyfikować dane, co zmniejsza potrzebę aktualizacji wag.

  Stabilizacja: Po kilku epokach (około 7-10) liczba aktualizacji osiąga zero. To wskazuje, że model nauczył się poprawnie klasyfikować wszystkie próbki w zbiorze treningowym (dla danych liniowo separowalnych).

  Efektywność uczenia: Szybkie zmniejszenie liczby błędów na początku oznacza, że przyjęta wartość współczynnika uczenia (eta=0.1) oraz liczba epok (n_iter=10) są odpowiednie dla tego problemu.

  Dane liniowo separowalne: Ponieważ liczba błędów osiąga zero, można przypuszczać, że zbiór danych jest liniowo separowalny.

  - Metoda "plot_decision_regions"

    Najpierw definiujemy liczbę barw (colors) i znaczników (markers), a następnie tworzymy mapę kolorów z listy barw za pomocą klasy ListedColormap. Następnie określamy minimalne i maksymalne wartości dwóch cech, które wykorzystujemy do wygenerowania siatki współrzędnych, tworząc tablice xx1 i xx2 za pomocą funkcji meshgrid.

    Ponieważ klasyfikator został wytrenowany na dwóch wymiarach cech, konieczna jest modyfikacja tablic xx1 i xx2 oraz utworzenie macierzy o takiej samej liczbie kolumn, jak zbiór uczący. Dzięki temu możemy zastosować metodę predict do przewidywania etykiet klas dla poszczególnych elementów siatki.

        plot_decision_regions(X, y, classifier=ppn)
        plt.xlabel("BallArea [px]")
        plt.ylabel("VoidArea [px]")
        plt.legend(loc='upper left')
        plt.show()
    <img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/decisionPlot.png?raw=true" alt="perceptron" width="800" >


## 3. Trenowanie modelu / TensorFlow 2

Skrypt implementuje model klasyfikacji obrazów binarnych przy użyciu frameworków TensorFlow i Keras.

Wybrałem TensorFlow w wersji > 2.0, ponieważ jest stabilniejsza. TensorFlow 1.0 Również wykorzystuje grafy statyczne, ale w wersji > 2.0 także dynamiczne grafy obliczeniowe, przez co uzyskujemy większą elastyczność. 


Implementuje architekturę Convolutional Neural Network (CNN) do klasyfikacji obrazów binarnych.
Automatyzuje ładowanie danych i wstępne przetwarzanie przy użyciu narzędzi Keras.
Zapisuje wyszkolony model do ponownego użycia.
Proces przewidywania klasy obrazów testowych.

Funkcjonalność:

#### 1. Importowanie bibliotek
Biblioteka:

- TensorFlow/Keras: Do budowania i trenowania sieci neuronowej.
- Pillow (PIL): Do wstępnego przetwarzania danych obrazu.
- NumPy: Do operacji numerycznych.
- os: Do obsługi operacji na plikach.

W niniejszym projekcie nie korzystam z obsługi procesora graficznego, w związku z czym nie instaluje biblioteki "tensorflow-gpu"

#### 2. Przygotowanie danych obrazu
treningowe obrazów:
Skrypt zmienia rozmiar wszystkich obrazów treningowych na 150x150 pikseli (img_width, img_height).
Obrazy są przeskalowywane poprzez podzielenie wartości pikseli przez 255 w celu ich normalizacji między [0, 1] przy użyciu ImageDataGenerator.
Metoda flow_from_directory ładuje obrazy i automatycznie je etykietuje na podstawie struktury folderów.

        train_data = train_rescale.flow_from_directory(
        'train/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
        )

#### 3. Architektura modelu
Model opiera się na architekturze sieci neuronowej splotowej (CNN) z następującymi warstwami:

- Warstwy splotowe (Conv2D): Wyodrębnij cechy przestrzenne z obrazów.
- Warstwy aktywacji: Użyj funkcji aktywacji ReLU, aby wprowadzić nieliniowość.
- Warstwy MaxPooling: Zmniejsz wymiary przestrzenne, zachowując ważne cechy.
- Warstwa spłaszczania: Spłaszcza mapy cech do tablicy 1D.
- Warstwy gęste: W pełni połączone warstwy do klasyfikacji, z dodanym dropoutem, aby zapobiec nadmiernemu dopasowaniu.
- Warstwa wyjściowa: Pojedynczy neuron z sigmoidalną funkcją aktywacji do klasyfikacji binarnej.

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

#### 4. Kompilacja modelu
Model jest kompilowany z następującymi konfiguracjami:

- Funkcja straty: binary_crossentropy, odpowiednia do klasyfikacji binarnej.
- Optymalizator: rmsprop, optymalizator oparty na gradiencie.
- Metryki: Śledzi dokładność podczas treningu.

        model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
        )

#### 5. Trening modelu
Model jest trenowany przez 128 epok przy użyciu danych treningowych (train_data).
Każda epoka przetwarza wszystkie partie treningowe zdefiniowane przez steps_per_epoch.

        model.fit(
        train_data,
        steps_per_epoch=len(train_data),
        epochs=128
        )

<img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/Epochs.png?raw=true" alt="Epochs" width="400" >

#### 6. Zapisywanie modelu
Zapis wytrenowanego modelu w dwóch formatach:

- Plik zawierający tylko wagi: Zapisany w formacie HDF5 (model_weights.weights.h5).
- Pełny model Keras: Zapisany w natywnym formacie .keras w celu ponownego użycia (model_keras.keras).

        model.save_weights('model_weights.weights.h5')
        model.save('model_keras.keras')
#### 7. Plot'y dokładności uczenia i straty uczenia

<img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/TrainingLossAndAccuracy.png?raw=true" alt="TrainingLossAndAccuracy" width="800" >

1. Wykres strat treningowych (Loss Plot)
Opis: Przedstawia, jak strata (funkcja kosztu, np. cross-entropy) zmienia się w trakcie epok treningowych.
Oś X: Numer epoki (Epoch) – odpowiada kolejnym iteracjom, w których model przechodzi przez cały zestaw danych treningowych.
Oś Y: Wartość straty – miara błędu modelu, gdzie niższa wartość oznacza lepsze dopasowanie modelu do danych.
Linia: Reprezentuje zmieniającą się wartość straty dla zestawu treningowego w każdej epoce.
Cel: Wartość straty powinna systematycznie maleć w trakcie treningu, co wskazuje, że model staje się bardziej precyzyjny w dopasowywaniu się do danych.


2. Wykres dokładności treningowej (Accuracy Plot)
Opis: Przedstawia zmieniającą się dokładność klasyfikacji (lub innej miary trafności) podczas treningu.
Oś X: Numer epoki (Epoch).
Oś Y: Dokładność (Accuracy) – miara, jak często model poprawnie przewiduje wyniki. Wyrażana jako ułamek lub procent (wartości od 0 do 1 lub 0% do 100%).
Linia: Pokazuje, jak dokładność modelu zmienia się w czasie.
Cel: Wartość dokładności powinna rosnąć, co oznacza, że model coraz lepiej przewiduje dane.

#### 8. Testowa prognoza obrazu
Ładowanie obrazów z katalogu test/, wstępnie przetwarzanie każdego obrazu (zmieniając rozmiar, normalizując i rozszerzając wymiary) i użycie wytrenowanego modelu do tworzenia prognoz.

- Wstępne przetwarzanie: Upewnienie się, że obrazy testowe pasują do wymiarów i skali użytych w treningu.
- Prognoza: Jeśli model przewiduje prawdopodobieństwo (result[0][0]) większe lub równe 0,5, obraz jest klasyfikowany jako „pass”; w przeciwnym razie jest to „fail”.

dla obrazu w test_images:

        img = Image.open('test/' + image).convert('RGB')
        result = model.predict(img)
        prediction = 'pass' if result[0][0] >= 0.5 else 'fail'
        print(f"Obraz {image} jest: {prediction}")

<img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/test.png?raw=true" alt="test" width="400" >

#### 9. Dane wyjściowe
Wynik predykcji dla każdego obrazu testowego w formacie:
Obraz 'nazwa pliku' jest: 'status'

## 4. Przewidywanie

Ten skrypt Pythona tworzy GUI przy użyciu tkinter do klasyfikacji obrazów z wstępnie wytrenowanym modelem Keras. Użytkownicy mogą przesłać obraz .png, który jest wyświetlany i wstępnie przetwarzany (zmieniony rozmiar, znormalizowany i wsadowy) przed przekazaniem do modelu w celu prognozowania. Wynik, „Pass” lub „Fail” (klasyfikacja binarna oparta na progu 0,5), jest wyświetlany w GUI. Używa Pillow do obsługi obrazów i tensorflow.keras do prognozowania modelu.

<img src="https://github.com/Mat3u52/BallGridArrayDefectDetectionByML/blob/main/examples/prediction.png?raw=true" alt="prediction" width="400" >

#### 1. Importowanie bibliotek

Biblioteka:

- tkinter: GUI.
- TensorFlow/Keras: Do budowania i trenowania sieci neuronowej.
- Pillow (PIL): Do wstępnego przetwarzania danych obrazu.
- NumPy: Do operacji numerycznych.
- os: Do obsługi operacji na plikach.


---

## Bibliografia:
- Raschka, Sebastian, and Vahid Mirjalili. *Python Machine Learning and deep learning*. 3rd ed., Packt Publishing, 2019.
- Matthes, Eric. *Python Crash Course*: A Hands-On, Project-Based Introduction to Programming. 2nd ed., No Starch Press, 2019.

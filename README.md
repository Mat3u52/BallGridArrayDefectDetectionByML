# BallGridArrayDefectDetectionByML

### Rozpoznanie defektu typu void dla układu "Ball Grid Array" przy pomocy machine learning.

IPC-A-610, zatytułowany „Akceptowalność zespołów elektronicznych”, jest powszechnie uznawanym standardem w branży elektronicznej, który zapewnia kompleksowe kryteria akceptacji zespołów elektronicznych, w tym tych wykorzystujących komponenty Ball Grid Array (BGA).

Jeśli chodzi o BGA, IPC-A-610 zajmuje się kilkoma krytycznymi aspektami, aby zapewnić jakość montażu:

Jakość połączenia lutowanego: Norma określa akceptowalne warunki dla połączeń lutowanych BGA, skupiając się na takich czynnikach, jak przesunięcie kulki lutowniczej, kształt i obecność pustych przestrzeni. Na przykład IPC-A-610F definiuje, że kulka BGA jest akceptowalna dla klas 1, 2 i 3, jeśli jej powierzchnia pusta jest mniejsza niż 30% powierzchni kuli w 2D kontroli rentgenowskiej.

Inspekcja i testowanie: Norma określa zalecane techniki inspekcji, takie jak analiza rentgenowska, w celu oceny integralności połączeń lutowanych BGA i wykrywania potencjalnych wad, takich jak pustki lub nieprawidłowe ustawienie.

Przestrzeganie kryteriów IPC-A-610 dla zespołów BGA pomaga zapewnić niezawodność i wydajność produktu, co jest szczególnie ważne w zastosowaniach wymagających wysokiej jakości zespołów elektronicznych.

PCB:
<img src="https://static.wixstatic.com/media/74cd3b_773e6cd81fb1433e972e27dbf33c53bd~mv2_d_4000_3000_s_4_2.jpg" alt="PCB" width="200">

BGA:
<img src="https://www.fs-pcba.com/wp-content/uploads/2022/12/1-9.jpg" alt="BGA" width="200">

Ball:
<img src="https://epp-europe-news.com/wp-content/uploads/3/6/3603682.jpg" alt="Ball" width="200">



## 1. Przygotowanie zestawu danych
- Informacje na temat przygotowanego wcześniej pliku .zip. Paczka zawiera zestaw różnych zdjęć ATG układu BGA po montażu w formacie .png 
  - Metoda "print_zip_summary":
  
          Podsumowanie zawartości pliku ZIP.
          Po podaniu ścieżki do pliku ZIP ta funkcja otwiera plik w trybie tylko do odczytu,
          pobiera listę wszystkich plików w nim zawartych i wyświetla podsumowanie pliku ZIP.
          
          W szczególności wyświetla:
            Nazwę bazową pliku ZIP. 
            Rozmiar pliku archiwum ZIP w bajtach.
            Całkowitą liczbę plików zawartych w archiwum.
            Listę ostatnich 5 plików w archiwum. Jeśli archiwum ZIP zawiera mniej niż 5 plików, wydrukuje je wszystkie.
      
  - Metoda "unzip_precess":
        
        Wypakowuje pliki z archiwum ZIP do określonego katalogu, pokazując postęp wypakowywania.
  
  - Metoda "tree":
  
        Rekurencyjnie wyświetla zawartość katalogu w ustrukturyzowanym, hierarchicznym formacie.
  
  - Metoda "show_samples":
  
        Losowo wybiera i wyświetla pięć obrazów .png z danego katalogu. Każdy obraz jest zmieniany na
        300x300 pikseli i wyświetlany w oknie pop-up przez 5 sekund. Na ten moment zestaw danych jest bardzo "zanieczyszczony" i nadmiarowy.

  - Metoda "recognizer":

        Przetwarza obraz w celu zidentyfikowania największego okrągłego obiektu, 
        oblicza jego średnicę i powierzchnię oraz podświetla puste obszary w obrębie marginesu wokół wykrytego konturu.
        Funkcja zostaje wywołana w pętli 'for' w celu odseparowanie zdjęć z defektem i bez defektu. 
        Dodatkowo w katalogu 'bitmapDiagnostics' umieszczone zostają pliki diagnostyczne.

## 2. Analiza danych (conda - notebook)
### regresja liniowa

## 3. Trenowanie modelu / TensorFlow 2

Skrypt implementuje model klasyfikacji obrazów binarnych przy użyciu frameworków TensorFlow i Keras.

Wybrałem TensorFlow w wersji > 2.0, ponieważ jest stabilniejsza. Również wykorzystuje grafy statyczne, ale w wersji > 2.0 także dynamiczne grafy obliczeniowe, przez co uzyskujemy większą elastyczność. 



Implementuje architekturę CNN do klasyfikacji obrazów binarnych.
Automatyzuje ładowanie danych i wstępne przetwarzanie przy użyciu narzędzi Keras.
Zapisuje wyszkolony model do ponownego użycia.
Proces przewidywania klasy obrazów testowych.

Funkcjonalność:

#### 1. Importowanie bibliotek
Bibliotek:

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

#### 6. Zapisywanie modelu
Zapis wytrenowanego modelu w dwóch formatach:

- Plik zawierający tylko wagi: Zapisany w formacie HDF5 (model_weights.weights.h5).
- Pełny model Keras: Zapisany w natywnym formacie .keras w celu ponownego użycia (model_keras.keras).

        model.save_weights('model_weights.weights.h5')
        model.save('model_keras.keras')

#### 7. Testowa prognoza obrazu
Ładowanie obrazów z katalogu test/, wstępnie przetwarzanie każdego obrazu (zmieniając rozmiar, normalizując i rozszerzając wymiary) i użycie wytrenowanego modelu do tworzenia prognoz.

- Wstępne przetwarzanie: Upewnienie się, że obrazy testowe pasują do wymiarów i skali użytych w treningu.
- Prognoza: Jeśli model przewiduje prawdopodobieństwo (result[0][0]) większe lub równe 0,5, obraz jest klasyfikowany jako „pass”; w przeciwnym razie jest to „fail”.

dla obrazu w test_images:

        img = Image.open('test/' + image).convert('RGB')
        result = model.predict(img)
        prediction = 'pass' if result[0][0] >= 0.5 else 'fail'
        print(f"Obraz {image} jest: {prediction}")

#### 8. Dane wyjściowe
Wynik predykcji dla każdego obrazu testowego w formacie:
Obraz 'nazwa pliku' jest: 'status'

## 4. Przewidywanie

Ten skrypt Pythona tworzy GUI przy użyciu tkinter do klasyfikacji obrazów z wstępnie wytrenowanym modelem Keras. Użytkownicy mogą przesłać obraz .png, który jest wyświetlany i wstępnie przetwarzany (zmieniony rozmiar, znormalizowany i wsadowy) przed przekazaniem do modelu w celu prognozowania. Wynik, „Pass” lub „Fail” (klasyfikacja binarna oparta na progu 0,5), jest wyświetlany w GUI. Używa Pillow do obsługi obrazów i tensorflow.keras do prognozowania modelu.

#### 1. Importowanie bibliotek

Bibliotek:

- tkinter: GUI.
- TensorFlow/Keras: Do budowania i trenowania sieci neuronowej.
- Pillow (PIL): Do wstępnego przetwarzania danych obrazu.
- NumPy: Do operacji numerycznych.
- os: Do obsługi operacji na plikach.


---

## Bibliografia:
- Raschka, Sebastian, and Vahid Mirjalili. *Python Machine Learning and deep learning*. 3rd ed., Packt Publishing, 2019.

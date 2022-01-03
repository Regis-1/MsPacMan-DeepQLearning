# MsPacMan-DeepQLearning
Repozytorium zawierające w sobie kody programów i narzędzie wykorzystanych podczas pisania pracy inżynierskiej pt. _"Narzędzie wspierające gracza w wybranej grzewideo, oparte o techniki uczenia maszynowego"_.
### Struktura repozytorium
- log2csv - narzędzie tworzące wygodny do analizy plik csv z logów powstałych w trakcie trenowania modelu
- train - folder zawierający program służący do trenowania modelu sieci neuronowej
- play - folder zawerający program służący do wspomagania gracza w grze _Ms. Pac\-Man_ wykorzystując do tego podany model
### Przygotowanie do uruchomienia
Cały projekt napisany jest w języku skryptowym Python, dlatego niewymagana jest jakakolwiek dodatkowa instalacja na urządzeniu poza instalacją wykorzystywanych modułów i bibliotek wspomnianego języka programowania. W pliku _requirements.txt_ zawarta jest lista wszystkich zastosowanych bibliotek zainstalowanych w środowisku Anaconda. Zalecane jest odtworzenie tego projektu również w tym środowisku, jednak nie jest to niezbędne. Przykład odtworzenia środowiska:
```
git clone https://github.com/Regis-1/MsPacMan-DeepQLearning
cd MsPacMan-DeepQLearning
conda create --name <env> --file requirements.txt
```
### Uruchomienie skryptów
Oba, glówne foldery (_train_ oraz _play_) są osobnymi programami ze wszystkimi innymi plikami. Aby uruchomić działanie każdego z nich wystarczy uruchomić moduł _main.py_ w wybranym folderze. Przykład:
```
cd play
python main.py
```

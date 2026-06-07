# WKIRO Project - LSTM dla danych motion capture

Projekt klasyfikuje sekwencje motion capture z datasetu chodu i biegu. Pipeline obejmuje:

- wczytanie plików `Post_Process/*.csv`,
- centrowanie markerów względem miednicy (`LASI`, `RASI`, `LPSI`, `RPSI`),
- standaryzację cech uczoną tylko na części treningowej,
- podział na okna czasowe,
- trening modeli LSTM dla trzech zadań,
- ewaluację predykcji na poziomie okien i całych nagrań.

## Zadania klasyfikacji

Kod obsługuje trzy zadania:

- `gait_type` - rozpoznanie typu ruchu, np. `walk_slow`, `walk_fast`, `run_comfortable`.
- `sex` - rozpoznanie płci uczestnika na podstawie ruchu.
- `participant_id` - identyfikacja konkretnego uczestnika.

Domyślny split to `within_participant`: każdy uczestnik jest reprezentowany w `train`, `val` i `test`, a jego próby są dzielone proporcjonalnie. To jest potrzebne zwłaszcza dla `participant_id`, ponieważ klasy uczestników muszą występować również w treningu.

Dla zadań `gait_type` i `sex` można eksperymentalnie użyć `participant_holdout`, gdzie całe osoby trafiają tylko do jednego zbioru. Nie używaj tego dla `participant_id`.

## Instalacja

Uruchom PowerShell w katalogu projektu:

```powershell
cd C:\Users\kwnuk\Documents\Studia\WKiRO\WKIRO-Project
```

Jeżeli środowisko `.venv` już istnieje, zainstaluj zależności:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Jeżeli trzeba utworzyć środowisko od zera:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Uwaga: TensorFlow na natywnym Windowsie zwykle użyje CPU. To normalne dla TensorFlow >= 2.11.

## Szybka weryfikacja danych

Możesz sprawdzić, czy loader widzi dane i metadane:

```powershell
.\.venv\Scripts\python.exe -c "import sys; from pathlib import Path; sys.path.append(str(Path('src').resolve())); from data_loading import discover_trials; from metadata import load_subject_metadata; print(len(discover_trials())); print(load_subject_metadata()[['participant_id','session','sex']].head())"
```

Oczekiwany wynik to około `3026` prób CSV i metadane uczestników z kolumną `sex`.

## Trening modeli

### Szybki trening testowy

Ten wariant ogranicza liczbę okien w każdym zbiorze, więc nadaje się do sprawdzenia, czy wszystko działa:

```powershell
.\.venv\Scripts\python.exe scripts\train_lstm_experiments.py --tasks gait_type sex participant_id --window-sizes 50 100 --lstm-units 32 64 64,32 --epochs 20 --max-windows-per-split 8000
```

### Pełniejszy trening

Ten wariant nie ogranicza liczby okien:

```powershell
.\.venv\Scripts\python.exe scripts\train_lstm_experiments.py --tasks gait_type sex participant_id --window-sizes 50 100 --lstm-units 32 64 64,32 --epochs 30
```

### Przykład tylko dla jednego zadania

```powershell
.\.venv\Scripts\python.exe scripts\train_lstm_experiments.py --tasks gait_type --window-sizes 100 --lstm-units 64 --epochs 30
```

### Ważne parametry

- `--tasks` - lista zadań: `gait_type`, `sex`, `participant_id`.
- `--window-sizes` - długości okien w klatkach, np. `50 100`.
- `--step-size` - przesunięcie okna; domyślnie `50`.
- `--lstm-units` - konfiguracje LSTM; `64,32` oznacza dwie warstwy.
- `--dropouts` - wartości dropout, np. `0.2 0.3`.
- `--learning-rates` - learning rate, np. `0.001 0.0005`.
- `--epochs` - maksymalna liczba epok.
- `--patience` - early stopping patience.
- `--max-windows-per-split` - limit okien na split; zostaw bez tego parametru dla finalnej ewaluacji.
- `--split-strategy` - `within_participant` albo `participant_holdout`.

## Notebook

Interaktywny wariant treningu znajduje się w:

```text
notebooks/03_experiments.ipynb.py
```

Plik ma komórki `# %%`, więc można go uruchamiać krok po kroku w VS Code albo PyCharm. Wyniki notebooka zapisują się w:

```text
models/lstm_experiments_notebook/
```

## Wyniki treningu

Domyślny skrypt zapisuje wyniki do:

```text
models/lstm_experiments/
```

Struktura katalogu:

```text
models/lstm_experiments/
  gait_type/
    split_summary.csv
    trial_split.csv
    standardizer.json
    window_100_step_50/
      train_windows.csv
      val_windows.csv
      test_windows.csv
      lstm_64_win_100_step_50_drop_0.3_lr_0.001/
        best_model.keras
        last_model.keras
        history.csv
        metrics.json
        window_predictions.csv
        window_metrics.json
        window_confusion_matrix.csv
        window_confusion_matrix.png
        recording_predictions.csv
        recording_metrics.json
        recording_confusion_matrix.csv
        recording_confusion_matrix.png
        window_vs_recording_metrics.csv
```

## Ewaluacja

Każdy run zapisuje dwie warstwy ewaluacji:

- `window_*` - metryki liczone dla każdego okna czasowego osobno.
- `recording_*` - metryki liczone po agregacji okien do całego nagrania.

Agregacja nagrania działa tak:

1. Model zwraca prawdopodobieństwa klas dla każdego okna.
2. Dla jednego pliku/nagrania uśredniane są prawdopodobieństwa ze wszystkich jego okien.
3. Predykcją całego nagrania jest klasa z najwyższym średnim prawdopodobieństwem.

Najważniejsze pliki:

- `window_predictions.csv` - predykcje i prawdopodobieństwa dla każdego okna.
- `recording_predictions.csv` - predykcje po agregacji do całych nagrań.
- `window_metrics.json` - accuracy, precision, recall, F1 dla okien.
- `recording_metrics.json` - accuracy, precision, recall, F1 dla nagrań.
- `window_confusion_matrix.png` - confusion matrix dla okien.
- `recording_confusion_matrix.png` - confusion matrix dla całych nagrań.
- `window_vs_recording_metrics.csv` - bezpośrednie porównanie metryk okien i nagrań.

## Podsumowanie wielu eksperymentów

Po treningu uruchom:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_lstm_results.py --output-dir models\lstm_experiments --top-k 3
```

Skrypt zapisze:

```text
models/lstm_experiments/evaluation_summary.csv
models/lstm_experiments/best_runs_top3.csv
```

`evaluation_summary.csv` zawiera metryki dla wszystkich runów. `best_runs_top3.csv` pokazuje najlepsze konfiguracje per zadanie według `recording_f1_macro`.

## Zalecana kolejność pracy

1. Zainstaluj zależności:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2. Uruchom szybki trening testowy:

```powershell
.\.venv\Scripts\python.exe scripts\train_lstm_experiments.py --tasks gait_type --window-sizes 50 --lstm-units 32 --epochs 3 --max-windows-per-split 1000
```

3. Sprawdź, czy powstały pliki:

```text
models/lstm_experiments/gait_type/.../window_confusion_matrix.png
models/lstm_experiments/gait_type/.../recording_confusion_matrix.png
```

4. Uruchom pełniejszy trening:

```powershell
.\.venv\Scripts\python.exe scripts\train_lstm_experiments.py --tasks gait_type sex participant_id --window-sizes 50 100 --lstm-units 32 64 64,32 --epochs 30
```

5. Zbierz wyniki:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_lstm_results.py --output-dir models\lstm_experiments --top-k 3
```

6. Do raportu użyj:

- `evaluation_summary.csv`,
- `best_runs_top3.csv`,
- `window_vs_recording_metrics.csv`,
- `window_confusion_matrix.png`,
- `recording_confusion_matrix.png`.

## Interpretacja wyników

Porównuj przede wszystkim:

- `recording_accuracy` - skuteczność po zagłosowaniu/uśrednieniu okien dla całego nagrania.
- `recording_f1_macro` - dobra metryka przy nierównych klasach.
- `window_accuracy` - pokazuje, jak stabilne są lokalne predykcje okien.
- różnicę między `window_*` i `recording_*` - jeżeli nagrania są wyraźnie lepsze, model myli pojedyncze okna, ale całościowy sygnał jest stabilny.

Do finalnego raportu lepiej używać wyników bez `--max-windows-per-split`, bo wtedy ewaluacja obejmuje wszystkie okna testowe.

# WKIRO Project - klasyfikacja motion capture z LSTM

Projekt trenuje modele LSTM dla danych motion capture chodu i biegu. Pipeline obejmuje wczytanie plików CSV, centrowanie markerów względem miednicy, standaryzację, tworzenie okien czasowych, trening modeli i ewaluację wyników na poziomie okien oraz całych nagrań.

## Zakres modeli

Skrypty obsługują trzy zadania:

- `gait_type` - rozpoznawanie typu ruchu, np. `walk_slow`, `walk_fast`, `run_comfortable`.
- `sex` - rozpoznawanie płci uczestnika.
- `participant_id` - identyfikacja uczestnika.

Domyślny preset `presentation` trenuje 7 konfiguracji LSTM dla każdego zadania. Zestaw jest oparty o założenia z prezentacji: różne długości okna, różna głębokość LSTM, liczba neuronów 15/32/64 i Adam z learning rate `0.001`.

Domyślne konfiguracje:

| Nazwa | Window | Step | LSTM units | Dropout | Dense |
|---|---:|---:|---|---:|---|
| `win5_lstm15` | 5 | 5 | 15 | 0.20 | - |
| `win10_lstm15x15` | 10 | 5 | 15, 15 | 0.20 | - |
| `win20_lstm15x15x15` | 20 | 10 | 15, 15, 15 | 0.25 | - |
| `win20_lstm32` | 20 | 10 | 32 | 0.30 | - |
| `win50_lstm32` | 50 | 25 | 32 | 0.30 | - |
| `win50_lstm64` | 50 | 25 | 64 | 0.30 | - |
| `win100_lstm64x32_dense32` | 100 | 50 | 64, 32 | 0.30 | 32 |

Łącznie domyślny trening daje 21 modeli: 7 konfiguracji x 3 zadania.

## Split danych

Domyślny split to `within_participant`: próby każdego uczestnika są proporcjonalnie dzielone na `train`, `val` i `test`. Dzięki temu każde zadanie, także `participant_id`, ma reprezentację wszystkich klas w części treningowej i testowej.

Dla zadań `gait_type` i `sex` można eksperymentalnie użyć `participant_holdout`, ale nie należy używać go dla `participant_id`, bo wtedy test zawierałby klasy uczestników niewidziane podczas treningu.

## Instalacja

W katalogu projektu utwórz środowisko i zainstaluj zależności:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS:

```bash
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

TensorFlow na natywnym Windowsie zwykle działa na CPU. To wystarczy do uruchomienia pipeline'u, ale pełny trening może potrwać.

## Struktura danych

Skrypty zakładają strukturę:

```text
data/
  metadata.xlsx
  <participant_id>/
    Session1/
      ...
    Session2/
      ...
```

Wczytywane są pliki:

```text
data/<participant_id>/Session*/<protocol>/<trial>/Post_Process/*.csv
```

## Szybka kontrola danych

Windows:

```powershell
.\.venv\Scripts\python.exe -c "import sys; from pathlib import Path; sys.path.append(str(Path('src').resolve())); from data_loading import discover_trials; from metadata import load_subject_metadata; print(len(discover_trials())); print(load_subject_metadata()[['participant_id','session','sex']].head())"
```

Linux/macOS:

```bash
./.venv/bin/python -c "import sys; from pathlib import Path; sys.path.append(str(Path('src').resolve())); from data_loading import discover_trials; from metadata import load_subject_metadata; print(len(discover_trials())); print(load_subject_metadata()[['participant_id','session','sex']].head())"
```

## Trening

### Szybki test działania

Uruchamia po 2 konfiguracje na zadanie i ogranicza liczbę okien:

Windows:

```powershell
.\.venv\Scripts\python.exe scripts\train_lstm_experiments.py --preset quick --epochs 3 --max-windows-per-split 1000
```

Linux/macOS:

```bash
./.venv/bin/python scripts/train_lstm_experiments.py --preset quick --epochs 3 --max-windows-per-split 1000
```

### Domyślny trening projektowy

Uruchamia 7 modeli dla każdego z trzech zadań:

Windows:

```powershell
.\.venv\Scripts\python.exe scripts\train_lstm_experiments.py --preset presentation --epochs 30
```

Linux/macOS:

```bash
./.venv/bin/python scripts/train_lstm_experiments.py --preset presentation --epochs 30
```

### Trening wybranych zadań

```bash
python scripts/train_lstm_experiments.py --preset presentation --tasks gait_type sex --epochs 30
```

### Własny grid parametrów

Jeżeli chcesz zignorować preset i użyć własnych wartości:

```bash
python scripts/train_lstm_experiments.py --preset custom --tasks gait_type --window-sizes 20 50 100 --lstm-units 32 64 64,32 --dropouts 0.2 0.3 --learning-rates 0.001 --epochs 30
```

Uwaga: tryb `custom` tworzy iloczyn kartezjański podanych parametrów. Łatwo przypadkowo uruchomić bardzo dużo modeli.

## Wyniki treningu

Domyślny katalog wyników:

```text
models/lstm_experiments/
```

Dla każdego modelu powstaje katalog runu zawierający:

- `best_model.keras` - najlepszy model według `val_loss`.
- `last_model.keras` - model po ostatniej epoce.
- `history.csv` - historia treningu.
- `metrics.json` - pełne metryki runu.
- `window_predictions.csv` - predykcje dla okien.
- `recording_predictions.csv` - predykcje po agregacji okien do całych nagrań.
- `window_metrics.json` - metryki dla okien.
- `recording_metrics.json` - metryki dla całych nagrań.
- `window_confusion_matrix.csv` i `.png`.
- `recording_confusion_matrix.csv` i `.png`.
- `window_vs_recording_metrics.csv` - bezpośrednie porównanie metryk.

## Ewaluacja i wykresy zbiorcze

Po zakończeniu treningu uruchom:

Windows:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_lstm_results.py --output-dir models\lstm_experiments --top-k 3
```

Linux/macOS:

```bash
./.venv/bin/python scripts/summarize_lstm_results.py --output-dir models/lstm_experiments --top-k 3
```

Skrypt zapisuje:

- `evaluation_summary.csv` - wszystkie modele i ich metryki.
- `best_runs_top3.csv` - najlepsze modele per zadanie.
- `plots/overall_best_recording_accuracy.png` - najlepsze accuracy dla każdego zadania.
- `plots/overall_best_recording_f1_macro.png` - najlepsze F1 macro dla każdego zadania.
- `plots/<task>_model_comparison.png` - porównanie modeli w obrębie zadania.
- `plots/<task>_window_vs_recording.png` - porównanie okien i całych nagrań.
- `plots/<task>_hyperparameter_effects.png` - wpływ rozmiaru okna, architektury LSTM i dropout.

## Jak wybrać najlepszy model

Do wyboru modelu używaj przede wszystkim:

- `recording_f1_macro` - główna metryka rankingowa, dobra przy nierównych klasach.
- `recording_accuracy` - skuteczność dla całych nagrań.
- `window_f1_macro` i `window_accuracy` - stabilność predykcji na krótkich fragmentach.
- confusion matrix - informacja, które klasy są mylone.

Praktyczna kolejność analizy:

1. Otwórz `evaluation_summary.csv` i posortuj po `recording_f1_macro`.
2. Sprawdź `best_runs_top3.csv`.
3. Porównaj `plots/<task>_model_comparison.png`.
4. Sprawdź `plots/<task>_hyperparameter_effects.png`, żeby opisać wpływ parametrów.
5. Dla najlepszego modelu obejrzyj `recording_confusion_matrix.png`.
6. Jeżeli `window_*` jest dużo gorsze niż `recording_*`, opisz, że pojedyncze okna bywają niestabilne, ale agregacja całego nagrania poprawia decyzję.

## Notebook

Interaktywny wariant eksperymentów znajduje się w:

```text
notebooks/03_experiments.ipynb.py
```

Plik ma komórki `# %%`, więc można go uruchamiać krok po kroku w edytorach obsługujących notebook-style Python files.

## Minimalny workflow

```bash
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/train_lstm_experiments.py --preset quick --epochs 3 --max-windows-per-split 1000
./.venv/bin/python scripts/summarize_lstm_results.py --output-dir models/lstm_experiments
```

Na Windowsie zamień `./.venv/bin/python` na:

```powershell
.\.venv\Scripts\python.exe
```

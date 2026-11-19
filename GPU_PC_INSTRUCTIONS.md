# Running Hadi's experiments on the RTX 5090

Thank you for lending the GPU. This page has everything needed: a one-time setup, then one command per night.

## What this is

A university research project (COE749, LAU) on the safety of open AI language models. Each run loads an open
model, removes its safety behaviour **in memory only** (nothing modified is saved), and measures how it responds.

- **The model's raw answers** are saved in `results\...\raw\`. Some of them answer harmful requests, so please
  don't open or share that folder.
- **Each run ends with a zip file for Hadi** that contains everything, those raw answers included. Please send it
  to him **privately** (a Google Drive link shared only with him, or WeTransfer), never in a group chat.
- A run uses the GPU for about **1 to 2 hours**. The first run downloads about **30 GB** of models.

## One-time setup (30 to 60 minutes, mostly downloading)

1. Install **Git for Windows**: https://git-scm.com/download/win (keep all the default options).
2. Open **PowerShell** (Start menu, type `PowerShell`).
3. Install **uv**, a Python installer. Paste this line, press Enter, then **close and reopen PowerShell**:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
4. Download the project into a folder with a short name, `abl` (long folder paths can break the install on Windows):
   ```powershell
   cd $HOME
   git clone https://github.com/hadi-soubra/LLM-model-abliteration-trap-research.git abl
   cd abl
   ```
5. Install Python and everything the project needs (a few minutes):
   ```powershell
   uv venv --python 3.12 .venv
   uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
   uv pip install --python .venv\Scripts\python.exe -e .
   ```
6. Log in to Hugging Face, where the models are downloaded from. Hadi will send you a **read-only token**.
   Paste it when asked, and answer `n` to "Add token as git credential?":
   ```powershell
   .venv\Scripts\hf auth login
   ```
7. Check that everything is ready:
   ```powershell
   $env:PYTHONUTF8 = "1"
   .venv\Scripts\python tools\preflight.py --config configs\gpu_qwen7b.yaml
   ```
   The last line should say **`ALL GOOD: ready for the night run.`**
   If it says `FIX THESE FIRST`, send Hadi a screenshot of the output.
8. Download the models now, so the night run does not have to (about 30 GB; progress bars show the speed):
   ```powershell
   .venv\Scripts\hf download Qwen/Qwen2.5-7B-Instruct
   .venv\Scripts\hf download meta-llama/Llama-Guard-3-1B
   .venv\Scripts\hf download meta-llama/Llama-Guard-3-8B
   ```
   If the last command says **`Access denied ... requires approval`**, that is fine: skip it, and the run uses the
   smaller judge.

## Before each night run

- **Stop the PC from sleeping:** Settings > System > Power > set *Sleep* to **Never** while plugged in.
- **Pause Windows Update for the night:** Settings > Windows Update > *Pause updates*. An update restart stops the run.

You can change both back afterwards.

## Each night run

Hadi will tell you which script to run that night (night 1 is below).

```powershell
cd $HOME\abl
git pull
$env:PYTHONUTF8 = "1"
.venv\Scripts\python steps\night1_baseline.py --config configs\gpu_qwen7b.yaml
```

- Leave the PowerShell window open. It prints progress lines with the time.
- **Don't click inside the PowerShell window while it runs.** Windows pauses a program while text in its window is
  selected. If the time on the progress lines stops moving, press `Esc` once and it continues.
- When it ends, it prints **`FINISHED.`** and the path of a zip file in the project folder, for example
  `SEND_TO_HADI_gpu_qwen7b_night1_full.zip` (a few hundred MB). Please send Hadi that file **privately**
  (Google Drive shared only with him, or WeTransfer).
- If that file is too big to send, send the small `..._summary.zip` next to it instead (numbers only).
- If something goes wrong it prints **`STOPPED`**: please send the same file anyway, it contains the log Hadi needs.

**If it gets interrupted** (power cut, Windows update, window closed), just run the same commands again.
Finished parts are skipped, and it continues where it stopped.

**To stop a run on purpose:** press `Ctrl+C` in the PowerShell window. (If some text is highlighted, press `Esc`
first, otherwise `Ctrl+C` only copies the text.)

## When the project is over

```powershell
.venv\Scripts\hf auth logout
```
Then delete the project folder, and the downloaded models in `C:\Users\<your name>\.cache\huggingface` (about 30 GB).

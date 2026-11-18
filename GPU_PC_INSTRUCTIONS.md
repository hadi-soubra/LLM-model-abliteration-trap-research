# Running Hadi's experiments on the RTX 5090

Thank you for lending the GPU. This page has everything needed: a one-time setup, then one command per night.

## What this is

A university research project (COE749, LAU) on the safety of open AI language models. Each run loads an open
model, removes its safety behaviour **in memory only** (nothing modified is saved), and measures how it responds.

- **The model's raw answers stay on your PC** in `results\...\raw\`. Some of them answer harmful requests,
  so please don't open or share that folder.
- **Only a small zip file with numbers goes back to Hadi.**
- A run uses the GPU for about **1 to 2 hours**. The first run downloads about **30 GB** of models.

## One-time setup (30 to 60 minutes, mostly downloading)

1. Install **Git for Windows**: https://git-scm.com/download/win (keep all the default options).
2. Open **PowerShell** (Start menu, type `PowerShell`).
3. Install **uv**, a Python installer. Paste this line, press Enter, then **close and reopen PowerShell**:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
4. Download the project:
   ```powershell
   cd $HOME
   git clone https://github.com/hadi-soubra/LLM-model-abliteration-trap-research.git
   cd LLM-model-abliteration-trap-research
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

## Before each night run

**Stop the PC from sleeping.** Settings > System > Power > set *Sleep* to **Never** while plugged in. A sleeping PC
stops the run. You can change it back afterwards.

## Each night run

Hadi will tell you which script to run that night (night 1 is below).

```powershell
cd $HOME\LLM-model-abliteration-trap-research
git pull
$env:PYTHONUTF8 = "1"
.venv\Scripts\python steps\night1_baseline.py --config configs\gpu_qwen7b.yaml
```

- Leave the PowerShell window open. It prints progress lines with the time.
- When it ends, it prints **`FINISHED. Please send this file to Hadi:`** followed by the path of a zip file
  (for example `SEND_TO_HADI_gpu_qwen7b_night1.zip` in the project folder). Please send Hadi that file. It is small.
- If it prints **`STOPPED`**, please send the same zip anyway: it contains the log Hadi needs.

**If it gets interrupted** (power cut, Windows update, window closed), just run the same commands again.
Finished parts are skipped, and it continues where it stopped.

**To stop a run on purpose:** click the PowerShell window and press `Ctrl+C`.

## When the project is over

```powershell
.venv\Scripts\hf auth logout
```
Then delete the project folder, and the downloaded models in `C:\Users\<your name>\.cache\huggingface` (about 30 GB).

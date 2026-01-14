# 👾⚡ TITAN VIRAL STUDIO (V8.5 - CLOUD SYNC)

**Il sistema definitivo per la generazione di contenuti virali (Reels, Horror Stories) completamente gratuito.**

## 🔥 NUOVE FUNZIONALITÀ
1. **GOD MODE (Sora-Class)**: Integrazione di **CogVideoX-5B**. Qualità cinematografica gratuita.
2. **Remote Bridge**: Usa il tuo PC locale come telecomando, ma fai lavorare la GPU di Colab.
3. **Cloud Sync**: Modifica i file in locale e aggiorna Colab con un click, senza ricaricare nulla a mano.

## 🚀 MODALITÀ IBRIDA (LOCALE + COLAB) - CONSIGLIATA
Vuoi usare il software sul tuo PC, ma non hai una GPU potente?

1.  Apri `TITAN_VIRAL_LAB.ipynb` su Google Colab e avvialo.
2.  Copia l'URL pubblico (es. `https://xxxx.gradio.live`) che appare alla fine.
3.  Sul tuo PC, installa le dipendenze leggere:
    ```bash
    pip install gradio gradio_client
    ```
4.  Esegui il **Remote Bridge**:
    ```bash
    python titan_remote.py
    ```
5.  Incolla l'URL di Colab e goditi la potenza del Cloud sul tuo Desktop.

## ☁️ COME AGGIORNARE LO SCRIPT SU COLAB (CLOUD SYNC)
Se vuoi modificare il codice (es. aggiungere nuovi preset):

1.  Crea un **Repository GitHub** (nuovo e vuoto).
2.  Esegui `update_cloud.bat`. Ti chiederà l'URL del repo (solo la prima volta).
3.  Lo script invierà le tue modifiche a GitHub.
4.  Su **Google Colab**, incolla l'URL del tuo Repo nella prima cella.
5.  Riavvia la cella. Il sistema scaricherà automaticamente la tua ultima versione.

## 🛠️ ARCHITETTURA
- `titan_video_engine.py`: Motore video (CogVideoX, SVD, AnimateDiff).
- `titan_remote.py`: Client locale per il controllo remoto.
- `update_cloud.bat`: Strumento di sincronizzazione automatica.

By | Janus & Tesavek⚡️👾

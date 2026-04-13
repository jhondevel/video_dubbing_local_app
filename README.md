# Doblaje local de video sin OpenAI

Esta versión hace el proceso **sin API de OpenAI** y sin costo por uso.

## Qué hace

1. Recibe un video desde una página web local.
2. Transcribe el audio con `faster-whisper`.
3. Traduce el texto con modelos `MarianMT` de Hugging Face.
4. Genera la nueva voz con `Piper`.
5. Une el audio nuevo al video y deja un `.mp4` final.

## Limitaciones reales

- No hace lip-sync.
- No conserva música ni sonido ambiente del audio original.
- La primera ejecución descarga modelos y puede tardar mucho.
- La traducción local depende de que exista un modelo `Helsinki-NLP/opus-mt-IDIOMA1-IDIOMA2` o de que existan los dos modelos vía inglés.
- La voz final depende de que usted descargue una voz de Piper para el idioma destino.
- En un PC sin GPU puede ser lento.

## Requisitos

- Windows 10/11
- Python 3.13
- Espacio libre suficiente para modelos locales
- Internet solo para descargar modelos la primera vez

## Carpeta del proyecto

Abra PowerShell dentro de la carpeta del proyecto.

## Instalación

Ejecute esto exactamente:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
python verify_install.py
```

Si `verify_install.py` termina bien, arranque la app:

```powershell
uvicorn main:app --reload
```

Abra en el navegador:

```text
http://127.0.0.1:8000
```

## Descargar voces de Piper

Primero liste voces:

```powershell
python -m piper.download_voices
```

Después descargue la que necesite a `data/voices`.
Ejemplos:

```powershell
python -m piper.download_voices es_ES-sharvard-medium --data-dir data/voices
python -m piper.download_voices en_US-lessac-medium --data-dir data/voices
python -m piper.download_voices fr_FR-upmc-medium --data-dir data/voices
python -m piper.download_voices de_DE-thorsten-medium --data-dir data/voices
python -m piper.download_voices it_IT-riccardo-x_low --data-dir data/voices
python -m piper.download_voices pt_BR-edresson-medium --data-dir data/voices
```

El proyecto espera encontrar dos archivos por voz dentro de `data/voices`:

```text
NOMBRE_VOZ.onnx
NOMBRE_VOZ.onnx.json
```

## Uso

1. Abra `http://127.0.0.1:8000`
2. Suba el video.
3. Escriba el idioma destino, por ejemplo `es`, `en`, `fr`, `de`, `it`, `pt`.
4. Si quiere, escriba una voz Piper concreta.
5. Pulse **Crear video doblado**.

## Idiomas previstos en la interfaz

La normalización actual reconoce estos códigos y nombres comunes:

- `es`, `español`, `spanish`
- `en`, `inglés`, `english`
- `fr`, `francés`, `french`
- `de`, `alemán`, `german`
- `it`, `italiano`, `italian`
- `pt`, `portugués`, `portuguese`
- `ru`, `ruso`, `russian`
- `ja`, `japonés`, `japanese`
- `zh`, `chino`, `chinese`
- `ko`, `coreano`, `korean`
- `ar`, `árabe`, `arabic`
- `hi`, `hindi`
- `tr`, `turco`, `turkish`
- `pl`, `polaco`, `polish`
- `nl`, `neerlandés`, `dutch`
- `ca`, `catalán`, `catalan`

## Solución de problemas

### 1. `No encontré la voz de Piper`

Descargue la voz indicada con `python -m piper.download_voices ... --data-dir data/voices`

### 2. Error con `sentencepiece` o `torch`

Actualice pip primero:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. El trabajo falla

Abra el archivo de error dentro de la carpeta del trabajo:

```text
data/jobs/ID_DEL_TRABAJO/error.log
```

### 4. Quiero otra voz

Liste voces:

```powershell
python -m piper.download_voices
```

Luego descargue una y escriba ese nombre exacto en el formulario.

```powershell
python -m piper.download_voices es_AR-daniela-high --data-dir data/voices
```

## Estructura

```text
main.py
requirements.txt
verify_install.py
app/
  main.py
  pipeline.py
  config.py
  db.py
  lang.py
  templates/
data/
  uploads/
  jobs/
  outputs/
  voices/
```

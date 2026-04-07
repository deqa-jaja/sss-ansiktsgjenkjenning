# Ansiktsgjenkjenning og velkomstsystem

Et system som gjenkjenner kunder via kamera og hilser dem med lyd på norsk. Bygget for Raspberry Pi.

## Hvordan det fungerer

1. Kameraet tar bilder i sanntid
2. Ansikter blir sammenlignet med kjente bilder i `images/`-mappen
3. Gjenkjente personer får en rektangel rundt ansiktet med navnet sitt
4. Systemet sier "velkommen kjære kunde [navn]" via høyttaler

## Teknologier

- **Python 3.11** 
- **face_recognition** - ansiktsgjenkjenning med deep learning
- **OpenCV** - bildeprosessering og kamera
- **gTTS** - tekst-til-tale på norsk
- **pydub** - avspilling av lyd
- **Raspberry Pi** med libcamera

## Oppsett

Krever Raspberry Pi med kamera.

```bash
pip install -r requirements.txt
```

Legg bilder av kjente personer i `images/`-mappen. Filnavnet blir brukt som navn (f.eks. `ola.jpg` → "Ola").

## Kjøring

```bash
python fcrecog-sss.py
```

Trykk `q` for å avslutte.

## Filstruktur

```
├── fcrecog-sss.py     # Hovedprogram - kamera og visning
├── face_recognizer.py # Klasse for ansiktsgjenkjenning (OOP)
├── images/            # Bilder av kjente personer
├── sounds/            # Genererte lydfiler (lages automatisk)
└── requirements.txt
```

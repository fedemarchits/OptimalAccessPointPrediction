"""
DATA EXPLORATION SCRIPT
Obiettivo: Capire la struttura del dataset e visualizzare i dati per il modello baseline
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pandas as pd
from collections import Counter

# ============================================================================
# CONFIGURAZIONE PATHS
# ============================================================================

# MODIFICA QUESTO PATH con il tuo path assoluto
BASE_DIR = Path(r"C:\Users\andre\Desktop\AlmaMater\Indusrty\src\PopulationDataset")

JSON_FILE = BASE_DIR / "final_clustered_samples.json"
SATELLITE_DIR = BASE_DIR / "satellite_images"
DEM_DIR = BASE_DIR / "dem_height"
SEGMENTATION_DIR = BASE_DIR / "segmentation_land_use"
CLUSTER_DIR = BASE_DIR / "cluster_images"

# ============================================================================
# 1. CARICARE E ANALIZZARE IL JSON
# ============================================================================

print("=" * 80)
print("📂 CARICAMENTO JSON...")
print("=" * 80)

# Carica il JSON (potrebbe richiedere qualche secondo, è 2.5GB!)
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

print(f"✅ JSON caricato con successo!")
print(f"Tipo di dato: {type(data)}")

# Se è una lista
if isinstance(data, list):
    print(f"Numero di samples: {len(data)}")
    print(f"\n📊 Struttura del primo sample:")
    print(json.dumps(data[0], indent=2))
    
    # Estrarre tutte le chiavi presenti
    all_keys = set()
    for sample in data[:100]:  # Controlla i primi 100
        all_keys.update(sample.keys())
    print(f"\n🔑 Chiavi disponibili nei samples:")
    for key in sorted(all_keys):
        print(f"  - {key}")

# Se è un dizionario
elif isinstance(data, dict):
    print(f"Chiavi principali: {list(data.keys())}")
    print(f"\n📊 Struttura:")
    print(json.dumps({k: type(v).__name__ for k, v in data.items()}, indent=2))

print("\n" + "=" * 80)
# ============================================================================
# 2. ANALISI STATISTICA DELLE LABELS
# ============================================================================

print("📊 ANALISI LABELS (Popolazione Raggiungibile)")
print("=" * 80)

# Definiamo la chiave specifica richiesta
label_key = 'population_15min_walk'

try:
    if isinstance(data, list):
        # RIMOZIONE VALORI NONE: Filtra la lista originale mantenendo solo i campioni validi
        initial_count = len(data)
        data = [sample for sample in data if sample.get(label_key) is not None]
        removed_count = initial_count - len(data)
        
        if removed_count > 0:
            print(f"⚠️ Rimossi {removed_count} campioni perché '{label_key}' era None.")

        # Estrazione delle label dalla lista filtrata
        labels = [sample[label_key] for sample in data]
        labels = np.array(labels)
        
        print(f"✅ Label trovate con chiave: '{label_key}'")
        print(f"Numero totale di samples validi: {len(labels)}")
        print(f"\nStatistiche:")
        print(f"   Min: {np.min(labels):.2f}")
        print(f"   Max: {np.max(labels):.2f}")
        print(f"   Mean: {np.mean(labels):.2f}")
        print(f"   Median: {np.median(labels):.2f}")
        print(f"   Std: {np.std(labels):.2f}")
        
        # Percentili
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\nPercentili:")
        for p in percentiles:
            val = np.percentile(labels, p)
            print(f"   {p}%: {val:.2f}")
        
        # Plot distribuzione
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(labels, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Popolazione Raggiungibile')
        plt.ylabel('Frequenza')
        plt.title(f'Distribuzione {label_key}')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.hist(np.log1p(labels), bins=50, edgecolor='black', alpha=0.7, color='orange')
        plt.xlabel('Log(Popolazione + 1)')
        plt.ylabel('Frequenza')
        plt.title('Distribuzione (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.boxplot(labels)
        plt.ylabel('Popolazione')
        plt.title('Boxplot Labels')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ Grafico salvato: label_distribution.png")
        plt.show()
        
    else:
        print(f"⚠️ Il dataset non è una lista. Impossibile procedere con la chiave '{label_key}'.")
        
except Exception as e:
    print(f"⚠️ Errore nell'estrazione delle label: {e}")
    print(f"Assicurati che la chiave '{label_key}' esista nel JSON.")

print("\n" + "=" * 80)

# ============================================================================
# 3. ANALISI IMMAGINI DISPONIBILI
# ============================================================================
# ============================================================================
# 3. ANALISI IMMAGINI DISPONIBILI (Fix per formati TIFF)
# ============================================================================

print("🖼️ ANALISI IMMAGINI")
print("=" * 80)

# Definiamo le cartelle da analizzare
image_dirs = {
    'Satellite': SATELLITE_DIR,
    'DEM Height': DEM_DIR,
    'Segmentation': SEGMENTATION_DIR,
    'Cluster': CLUSTER_DIR
}

# Estensioni da cercare (inclusi i formati GeoTIFF del tuo dataset)
extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']

image_counts = {}

for name, path in image_dirs.items():
    if path.exists():
        # Cerchiamo tutti i file che corrispondono alle estensioni indicate
        found_images = []
        for ext in extensions:
            # glob è case-insensitive su Windows, ma usiamo una ricerca robusta
            found_images.extend(list(path.glob(ext)))
            found_images.extend(list(path.glob(ext.upper()))) # Per sicurezza su Linux/Mac
        
        # Rimuoviamo eventuali duplicati e contiamo
        unique_images = list(set(found_images))
        image_counts[name] = len(unique_images)
        
        status = "✅" if len(unique_images) > 0 else "⚠️ Vuota"
        print(f"{status} {name:15s}: {len(unique_images):6d} file trovati")
        
        # Mostriamo le estensioni trovate per debug
        if len(unique_images) > 0:
            found_exts = set(f.suffix.lower() for f in unique_images)
            print(f"    Estensioni rilevate: {found_exts}")
    else:
        image_counts[name] = 0
        print(f"❌ {name:15s}: Cartella non trovata al path {path}")

print("\n" + "=" * 80)
# ============================================================================
# 4. VISUALIZZAZIONE ESEMPI (Versione con Suffissi Specifici)
# ============================================================================

print("🎨 VISUALIZZAZIONE ESEMPI RANDOM")
print("=" * 80)

def load_and_show_sample(sample_idx=0):
    # 1. Prendiamo i file master dalla cartella Satellite
    satellite_files = list(SATELLITE_DIR.glob('*_10m.tif'))
    if not satellite_files:
        print("⚠️ Nessuna immagine satellitare trovata! Controlla il path.")
        return
    
    # 2. Identifichiamo la radice (es. da "Amadora_Portugal_10m.tif" estraiamo "Amadora_Portugal")
    example_file = satellite_files[sample_idx]
    # Rimuoviamo '_10m' e l'estensione per avere la base pulita
    base_name = example_file.name.replace('_10m.tif', '') 
    
    print(f"\n📍 Analisi City: {base_name}")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 3. Definiamo la mappatura esatta basata sui tuoi esempi
    # Nome Titolo | Cartella | Suffisso specifico | Estensione
    modalities = [
        ('Satellite RGB', SATELLITE_DIR,     '_10m',          '.tif'),
        ('DEM Height',    DEM_DIR,           '_DEM_10m',      '.tif'),
        ('Land Use',      SEGMENTATION_DIR,  '_LandUse_10m',  '.tif'),
        ('Cluster Map',   CLUSTER_DIR,       '_clusters',     '.png'),
    ]
    
    for idx, (title, folder, suffix, ext) in enumerate(modalities):
        ax = axes[idx]
        # Costruiamo il path finale: es. BASE_DIR / "dem_height" / "Amadora_Portugal" + "_DEM_10m" + ".tif"
        img_path = folder / f"{base_name}{suffix}{ext}"
        
        if img_path.exists():
            try:
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # Normalizzazione per la visualizzazione (per i DEM a 32-bit)
                display_img = img_array.copy()
                if display_img.dtype != np.uint8:
                    d_min, d_max = display_img.min(), display_img.max()
                    if d_max > d_min:
                        display_img = (display_img - d_min) / (d_max - d_min)
                
                # Visualizzazione con colori appropriati
                if 'DEM' in title:
                    ax.imshow(display_img, cmap='terrain')
                elif 'Land Use' in title:
                    ax.imshow(display_img, cmap='tab10') # Colori distinti per le classi
                else:
                    ax.imshow(display_img)
                
                ax.set_title(f"{title}\n{img_array.shape}")
                print(f"  ✅ {title:15s}: Caricato ({img_path.name})")
            except Exception as e:
                ax.text(0.5, 0.5, f'Errore: {e}', ha='center', va='center', fontsize=8)
                print(f"  ❌ {title:15s}: Errore caricamento -> {e}")
        else:
            ax.text(0.5, 0.5, 'NON TROVATO', ha='center', va='center', color='red')
            ax.set_title(f"{title}\n(missing)")
            print(f"  ⚠️ {title:15s}: Mancante ({img_path.name})")
            
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'city_sample_{sample_idx}.png', dpi=150)
    plt.show()

# Eseguiamo su 3 città a caso
num_cities = len(list(SATELLITE_DIR.glob('*_10m.tif')))
# if num_cities > 0:
#     indices = np.random.choice(num_cities, min(3, num_cities), replace=False)
#     for i in indices:
load_and_show_sample(3)
# ============================================================================
# 5. VERIFICA CORRISPONDENZA JSON <-> IMMAGINI
# ============================================================================

print("🔗 VERIFICA CORRISPONDENZA JSON ↔ IMMAGINI")
print("=" * 80)

# Questa parte dipende molto da come sono organizzati i dati
# Esempio: se nel JSON c'è un campo 'filename' o 'id'

if isinstance(data, list):
    print(f"Samples nel JSON: {len(data)}")
    print(f"Immagini satellitari: {image_counts['Satellite']}")
    
    # Prova a capire come matchare JSON e immagini
    print("\n💡 SUGGERIMENTO:")
    print("Verifica come sono collegati JSON e immagini:")
    print("- C'è un campo 'filename' nel JSON?")
    print("- C'è un campo 'id' o 'tile_id'?")
    print("- Le coordinate (lat/lon) sono usate come nome file?")
    
    # Mostra i campi del primo sample per aiutare
    if len(data) > 0:
        print("\nCampi del primo sample:")
        for key, value in data[0].items():
            print(f"  {key}: {type(value).__name__} = {value if not isinstance(value, (dict, list)) else '...'}")

print("\n" + "=" * 80)

# ============================================================================
# 6. CREARE UN DATAFRAME RIASSUNTIVO
# ============================================================================

print("📋 CREAZIONE DATAFRAME RIASSUNTIVO")
print("=" * 80)

# Prova a creare un DataFrame con le info principali
# ADATTA QUESTA PARTE in base alla struttura del tuo JSON

if isinstance(data, list) and len(data) > 0:
    try:
        # Estrai campi base (adatta in base al tuo JSON)
        df_data = []
        
        for i, sample in enumerate(data):
            row = {'index': i}
            
            # Aggiungi tutti i campi del sample (escludendo nested objects)
            for key, value in sample.items():
                if not isinstance(value, (dict, list)):
                    row[key] = value
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        print(f"✅ DataFrame creato con {len(df)} righe e {len(df.columns)} colonne")
        print(f"\nPrime 5 righe:")
        print(df.head())
        
        print(f"\nInfo colonne:")
        print(df.info())
        
        print(f"\nStatistiche descrittive:")
        print(df.describe())
        
        # Salva il dataframe
        df.to_csv('dataset_summary.csv', index=False)
        print(f"\n✅ DataFrame salvato: dataset_summary.csv")
        
    except Exception as e:
        print(f"⚠️ Errore nella creazione del DataFrame: {e}")

print("\n" + "=" * 80)

# ============================================================================
# 7. RACCOMANDAZIONI PER IL MODELLO BASELINE
# ============================================================================

print("🎯 RACCOMANDAZIONI PER IL MODELLO BASELINE")
print("=" * 80)

print("""
📌 COSA USARE PER IL BASELINE MODEL (Fase 1):

1. INPUT: 
   ✅ Immagini RGB Satellitari (satellite_images/)
   ❌ DEM, Segmentation, Cluster → aggiungi dopo!

2. LABEL:
   ✅ Popolazione raggiungibile (dal JSON)
   
3. PREPROCESSING NECESSARIO:
   - Normalizzazione immagini: dividere per 255 o standardizzazione ImageNet
   - Split geografico: train/val/test (70/15/15)
   - Verificare che ogni immagine abbia una label corrispondente

4. ARCHITETTURA BASELINE:
   ResNet18 (pretrained) → GAP → FC layers → Output (1 value)

5. PROSSIMI PASSI:
   a) Creare PyTorch Dataset class
   b) Implementare DataLoader
   c) Verificare matching JSON ↔ immagini
   d) Iniziare training!
""")

print("=" * 80)
print("✅ ESPLORAZIONE COMPLETATA!")
print("=" * 80)
print("\nFile generati:")
print("  - label_distribution.png")
print("  - sample_visualization_*.png")
print("  - dataset_summary.csv")
print("\n💡 Controlla questi file e poi adatta il codice in base alla struttura del tuo JSON!")
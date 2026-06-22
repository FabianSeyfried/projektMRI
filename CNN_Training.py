#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import nibabel as nib
import logging
import mri_utils
import utils
import sys

# torch.set_num_threads(24) # Anzahl deiner zugewiesenen Kerne
# torch.set_num_interop_threads(24)
print(torch.cuda.is_available())


# In[2]:


# Hyperparameter
AMOUNT_VOLUMES_TO_LOAD = 3
PATCH_SIZE = (32, 32, 31)
RUN_WITH_TRAINING = True
SAVE_MODEL = True
OVERLAP = 0.5
STOP_AFTER_TAINING = True


# ## Import Datasets, Add Degraded and Ground Truth

# In[3]:


# Funktion zum Laden und Degradieren eines Volumes
def process_volume(file_path):
    img = nib.load(file_path)

    volume = np.asanyarray(img.dataobj, dtype=np.float32)
    print(f"Loaded volume shape: {volume.shape}")

    # Degraded-Daten erstellen
    gt_vol, degraded_vol = mri_utils.generate_dataset_list(
        [volume],
        keep_fraction=0.6,
        noise_min=0.01,
        noise_max=0.05
    )

    return gt_vol[0], degraded_vol[0]


# In[4]:


import os
import nibabel as nib
import numpy as np
from tqdm import tqdm 
import tracemalloc

tracemalloc.start()  # Tracke RAM-Nutzung

# Konfiguration
data_folder = "/srv/fMRI-data/"
output_dir = "./processed_data"  # Verzeichnis für zwischengespeicherte Daten
os.makedirs(output_dir, exist_ok=True)


# Liste aller NIfTI-Dateien
nii_files = sorted([
    f for f in os.listdir(data_folder)
    if f.endswith(".nii.gz")
])

print(f"Found {len(nii_files)} NIfTI files in {data_folder}")

# Optional: Nur die ersten 5 Volumes verarbeiten, um die Funktion zu testen
nii_files = nii_files[:AMOUNT_VOLUMES_TO_LOAD]  


# Daten als NumPy-Arrays speichern (oder direkt verarbeiten)
gt : list[np.ndarray] = []
degraded : list[np.ndarray] = []

for file in tqdm(nii_files, desc="Processing files"):
    file_path = os.path.join(data_folder, file)

    # Volume laden und GT sowie degraded Daten erstellen
    gt_vol, degraded_vol = process_volume(file_path)
    gt.append(gt_vol)
    degraded.append(degraded_vol)

print("Daten erfolgreich verarbeitet und gespeichert.")

print(type(gt))
print(type(degraded))



utils.show_ram_usage()


# ## Erstelle Chunks

# In[5]:


def generate_flexible_patch_coordinates(volume_shape, patch_size, overlap_factor=0.5):
    """
    Generiert Patch-Koordinaten mit einem prozentualen Overlap (z.B. 0.5 für 50%).
    Garantiert, dass auch die Ränder (Edges) vollständig abgedeckt werden, indem am Rand etwas mehr Überlappung genutzt wird, anstatt Voxels auszuschließen.
    """
    x_dim, y_dim, z_dim = volume_shape
    x_patch, y_patch, z_patch = patch_size
    
    # Berechne den Stride dynamisch basierend auf dem gewünschten Overlap
    x_stride = max(1, int(x_patch * (1 - overlap_factor))) # 16 
    y_stride = max(1, int(y_patch * (1 - overlap_factor))) # 16 
    z_stride = max(1, int(z_patch * (1 - overlap_factor))) # 15
    
    # Generiere die Startpunkte und Endpunkte als Tupel
    def get_1d_coordinates(dim_size, patch_size, stride):
        starts = list(range(0, dim_size - patch_size + 1, stride))
        
        # WICHTIG: Wenn der letzte Patch nicht exakt am Rand aufhört,
        # füge manuell einen Patch hinzu, der exakt am Rand endet.
        if (dim_size - patch_size) not in starts:
            starts.append(dim_size - patch_size)
            
        return [(start, start + patch_size) for start in starts]

    # Start & Endpunkte für jede Achse berechnen
    x_coords = get_1d_coordinates(x_dim, x_patch, x_stride)
    y_coords = get_1d_coordinates(y_dim, y_patch, y_stride)
    z_coords = get_1d_coordinates(z_dim, z_patch, z_stride)
    
    # Alle Kombinationen (3D-Gitter) zusammenbauen
    coords = []
    for x_start, x_end in x_coords:
        for y_start, y_end in y_coords:
            for z_start, z_end in z_coords:
                coords.append(((x_start, x_end), (y_start, y_end), (z_start, z_end)))
                
    return coords


# In[6]:


"""
Iteriere über alle Volumes und Zeitschritte
Generiere 3D Patch-Koordinaten und speichere sie mit Volume- und Timestep-Info:
"""
print("Anzahl Volumes (gt):", len(gt))
coords = []

for volume_idx, volume in enumerate(gt):
    
    # OPTIMIERUNG: Da die räumliche Form (X, Y, Z) für alle Zeitschritte dieses 
    # Volumes gleich bleibt, berechnen wir die Basis-Koordinaten nur EINMAL pro Volume.
    coords_per_volume = generate_flexible_patch_coordinates(volume.shape[:3], PATCH_SIZE, overlap_factor=OVERLAP)
    
    timesteps = volume.shape[3]
    for t in range(timesteps):
        # Jetzt verknüpfen wir die berechneten 3D-Koordinaten mit dem aktuellen Zeitschritt
        # Eigentlich nur Anhängen des Volumes und des Timesteps an Koordinaten
        for (x_start, x_end), (y_start, y_end), (z_start, z_end) in coords_per_volume:
            coords.append([
                volume_idx, t, 
                x_start, x_end, 
                y_start, y_end, 
                z_start, z_end
            ])

# Konvertierung in ein performantes NumPy-Array
coords = np.array(coords, dtype=np.int32)

print("Finale Array-Form:", coords.shape) 
print("Beispiel-Koordinaten:")
print("Volume Idx, Timestep, X Start, X End, Y Start, Y End, Z Start, Z End:")


print("Erster Patch:", coords[0])
print("Letzter Patch:", coords[-1])

# Ohne Überlappung
# 1. Volume => 10512 Patches
# 10512 => 48 (Anzahl Patches per Bild: 4*4*3) * 219 timesteps
# Patches pro Bild: 128 / 32= 4 für X und Y, 93 / 31 = 3 für Z

# Durch Überlappung kommt es zu Versechsfachung der Patches. Patches pro Timestep jetzt 294 (7*7*6)
# 294 Patches pro Bild * 219 Timesteps => 64_386


# ### Visualize same Ground Truth and Degraded image

# In[7]:


# Visualisiere erstes volumes, erster patch
first_coords = coords[0]  # Koordinaten des ersten Volumes
Volume_Idx, Timestep, X_Start, X_End, Y_Start, Y_End, Z_Start, Z_End = first_coords

volume_gt = gt[Volume_Idx]
volume_dg = degraded[Volume_Idx]

print("Volume GT shape:", volume_gt.shape)
print("Volume DG shape:", volume_dg.shape)

# display volume from with width, heigth, 0 for 2d image and timestep
image_gt = volume_gt[X_Start:X_End, Y_Start:Y_End, 0, Timestep]
image_dg = volume_dg[X_Start:X_End, Y_Start:Y_End, 0, Timestep]


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image_gt)
plt.title("GT Patch")

plt.subplot(1,2,2)
plt.imshow(image_dg)
plt.title("DG Patch")

plt.show()


# ### Normalisierung

# In[8]:


def precompute_volume_stats(degraded_volumes, bg_threshold=10.0):
    """
    Berechnet Mittelwert und Standardabweichung für jedes Volumen und Timestep basierend auf einem Schwellwert für Hintergrundrauschen.
    Die Statistiken werden für die Normalisierung der Daten vor dem Training verwendet.
    """
    volume_stats = {}
    
    for vol_idx, deg_volume in enumerate(degraded_volumes):
        volume_stats[vol_idx] = {}
        num_timesteps = deg_volume.shape[-1]
        
        for t in range(num_timesteps):
            vol_3d = deg_volume[..., t]
            brain_mask = vol_3d > bg_threshold
            
            if np.sum(brain_mask) > 0:
                mean_val = float(np.mean(vol_3d[brain_mask]))
                std_val = float(np.std(vol_3d[brain_mask])) + 1e-8
            else:
                mean_val = float(np.mean(vol_3d))
                std_val = float(np.std(vol_3d)) + 1e-8
            
            volume_stats[vol_idx][t] = (mean_val, std_val)
            
    return volume_stats


calculated_stats = precompute_volume_stats(degraded, bg_threshold=10.0)
print(len(calculated_stats))


# ### Dataset

# In[9]:


class VolumeDataset(Dataset):
    def __init__(self, degraded_volumes, gt_volumes, coords, volume_stats):
        """
        Args:
            degraded_volumes: Liste/Array der degradierten fMRI-Daten (Input).
                              Erwartete Form pro Element: [X, Y, Z, Timesteps]
            gt_volumes:       Liste/Array der Ground-Truth-Daten (Target).
            coords:           NumPy-Array mit den Koordinaten (Gesamtanzahl_Patches, 8).
            bg_threshold:     Schwellenwert, um den dunklen Hintergrund von der 
                              Normalisierung auszuschließen.
        """
        assert len(degraded_volumes) == len(gt_volumes), "Anzahl der degraded und gt Volumes muss gleich sein!"
        
        self.degraded_volumes = degraded_volumes
        self.gt_volumes = gt_volumes
        self.coords = coords
        self.volume_stats = volume_stats

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # 1. Koordinaten auslesen
        coord = self.coords[idx]
        volume_idx, timestep, x_start, x_end, y_start, y_end, z_start, z_end = coord
        
        # 2. Volumes holen
        deg_volume = self.degraded_volumes[volume_idx]
        gt_volume  = self.gt_volumes[volume_idx]

        # 3. Patch erstellen
        patch_deg = deg_volume[x_start:x_end, y_start:y_end, z_start:z_end, timestep]
        patch_gt  = gt_volume[x_start:x_end, y_start:y_end, z_start:z_end, timestep]

        # 4. In PyTorch-Tensoren umwandeln (1, X, Y, Z)
        tensor_deg = torch.tensor(patch_deg, dtype=torch.float32).unsqueeze(0)
        tensor_gt  = torch.tensor(patch_gt, dtype=torch.float32).unsqueeze(0)

        # Statistiken heraussuchen (O(1) Komplexität, extrem schnell)
        mean_vol, std_vol = self.volume_stats[volume_idx][timestep]

        # 6. Normalisierung mit den 3D-Volume-Werten
        tensor_deg = (tensor_deg - mean_vol) / std_vol
        tensor_gt  = (tensor_gt - mean_vol) / std_vol  # Wichtig: Gleiche Werte für das Target!

        return tensor_deg, tensor_gt, coord, mean_vol, std_vol


# ### Dataset + Dataloader

# In[ ]:


from torch.utils.data import Subset, DataLoader
import numpy as np
import gc

# 1. Dataset initialisieren
dataset = VolumeDataset(degraded, gt, coords, calculated_stats)
print(f"Gesamtanzahl Patches im Dataset: {len(dataset)}")

# 2. Dynamischen Volume-basierten Split berechnen
total_volumes = len(gt)
print(total_volumes)

if total_volumes > 1:
    # Zuteilung: z.B. 80% der Volumina für Train, mindestens aber 1 für Test
    num_train_vols = max(1, int(0.8 * total_volumes))
    
    # Da die nii_files in In[4] sortiert geladen wurden, nehmen wir die ersten für Train
    train_vols = list(range(0, num_train_vols))
    test_vols = list(range(num_train_vols, total_volumes))
else:
    # Fallback, falls AMOUNT_VOLUMES_TO_LOAD = 1 zum Testen genutzt wird
    print("WARNUNG: Nur 1 Volume geladen. Split erfolgt testweise auf Timestep-Ebene!")
    train_vols = [0]
    test_vols = [0]



# 3. Indizes filtern (c[0] ist der volume_idx im coords-Array)
if total_volumes > 1:
    train_indices = [i for i, c in enumerate(coords) if c[0] in train_vols]
    test_indices  = [i for i, c in enumerate(coords) if c[0] in test_vols]
else:
    # Wenn nur 1 Volume existiert, splitten wir die Timesteps (c[1]), um einen Crash zu verhindern
    max_t = coords[:, 1].max()
    split_t = int(0.8 * max_t)
    train_indices = [i for i, c in enumerate(coords) if c[1] <= split_t]
    test_indices  = [i for i, c in enumerate(coords) if c[1] > split_t]



# 4. Subsets erstellen statt random_split
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

print(f"Train Volumes: {train_vols} | Patches: {len(train_dataset)}")
print(f"Test Volumes: {test_vols} | Patches: {len(test_dataset)}")

# 5. Dataloader unverändert lassen
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
test_dataloader  = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

# WICHTIG: Val-Dataloader nutzt weiterhin das KOMPLETTE Dataset für die Rekonstruktion
val_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# Garbage Collector
gc.collect()
utils.show_ram_usage()


# ###  3D CNN

# In[11]:


import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x + residual)


class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()

        # First block: input -> 64 channels
        self.first = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 7 repeated blocks
        self.blocks = nn.Sequential(
            *[ConvBlock(64) for _ in range(7)]
        )

        # Last block (no activation)
        self.last = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.blocks(x)
        x = self.last(x)
        return x


# In[ ]:


#optimizer
criterion_mse = nn.MSELoss()
criterion_l1  = nn.L1Loss()
model = CNN3D()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

#training setup
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ### Training & Test Routine

# In[13]:


# Durch die Überlappung dauert das Training pro Epoche nicht mehr 2min, sondern eher länger, da mehr Daten
from torch.amp import autocast, GradScaler

#training
if RUN_WITH_TRAINING:
    model = model.to(device)
    # für Visualisierung
    train_losses = []
    test_losses = []
    scaler = GradScaler() #

    print("Starting epochs...")
    for epoch in range(num_epochs):

        epoch_loss_train = 0
        epoch_loss_test = 0

        model.train()
        for dg_batch, gt_batch, coord_batch, mean_vol_batch, std_vol_batch in train_dataloader:
        
            dg_batch = dg_batch.to(device)
            gt_batch = gt_batch.to(device)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):

                output = model(dg_batch)

                loss_mse = criterion_mse(output, gt_batch)
                loss_l1 = criterion_l1(output, gt_batch)
                loss = 0.3 * loss_mse + 0.7 * loss_l1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss_train += loss.item()

        model.eval()
        for dg_batch, gt_batch, coord_batch, mean_vol_batch, std_vol_batch in test_dataloader:
            
            dg_batch = dg_batch.to(device)
            gt_batch = gt_batch.to(device)

            with torch.no_grad():
                pred = model(dg_batch)
                
                loss_mse = criterion_mse(pred, gt_batch)
                loss_l1 = criterion_l1(pred, gt_batch)
                loss = 0.3 * loss_mse + 0.7 * loss_l1

                epoch_loss_test += loss.item()

        # Loss wird zuvor aufsummiert. Bilde Mittel über Anzahl Batches, da sonst mehr Batches (bei Train) für höheren Loss sorgt 
        epoch_loss_train = epoch_loss_train / len(train_dataloader)
        epoch_loss_test = epoch_loss_test / len(test_dataloader)

        train_losses.append(epoch_loss_train)
        test_losses.append(epoch_loss_test)

        # Am Ende der Epoche den Scheduler updaten
        scheduler.step(epoch_loss_test)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss_train:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {epoch_loss_test:.4f}")



    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(test_losses, label="Test Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


# Modell speichern und laden - wichtig für NLM

# In[ ]:


if RUN_WITH_TRAINING and SAVE_MODEL:
    torch.save(model.state_dict(), 'train_test_split_improved30epochs.pth')


# In[ ]:


if not RUN_WITH_TRAINING:
    model = CNN3D() 

    # 2. Gewichte laden
    model.load_state_dict(torch.load('train_test_split_improved30epochs.pth'))

if STOP_AFTER_TAINING:
    sys.exit()


# ### Inference of Patches, Denormalize and Reconstruct into entire image again

# In[ ]:


def inference_and_reconstruction(model: CNN3D, gt: list, val_dataloader: DataLoader, device: torch.device):
    """
    Führt GPU-Inferenz durch und rekonstruiert normalisierte, überlappende Patches
    korrekt zurück in die ursprüngliche 4D-fMRI-Struktur mittels Averaging.
    """
    model = model.to(device)
    model.eval()

    # Listen für die rekonstruierten Volumina und die Counter-Zähler
    predictions = []
    count_volumes = []

    # Initialisiere die Ziel-Strukturen auf der CPU
    for volume in gt:
        vol_shape = volume.shape  # (X, Y, Z, Timesteps)
        predictions.append(torch.zeros(vol_shape, dtype=torch.float32))
        # Der Counter trackt, wie viele Patches ein Voxel überlagern
        count_volumes.append(torch.zeros(vol_shape, dtype=torch.float32))

    print(f"Prediction_volumes initialisiert: {len(predictions)} Volumes")
    print("Starte Inferenz, Denormalisierung und Overlap-Rekonstruktion...")

    with torch.no_grad():
        for dg_batch, _, coord_batch, mean_vol_batch, std_vol_batch in val_dataloader:
            
            # GPU Inferenz
            dg_batch = dg_batch.to(device)
            prediction_batch = model(dg_batch)
            prediction_batch = prediction_batch.cpu()  # Zurück auf CPU für RAM-schonende Rekonstruktion
            
            current_batch_size = prediction_batch.shape[0]

            # Reconstruct
            for i in range(current_batch_size):
                # Koordinaten extrahieren
                c = coord_batch[i].numpy().astype(int)
                volume_idx, timestep, x_start, x_end, y_start, y_end, z_start, z_end = c

                # Statistiken für diesen spezifischen Patch holen
                # (Sicherstellen, dass es sich um skalare Floats handelt)
                mean_vol = float(mean_vol_batch[i])
                std_vol = float(std_vol_batch[i])

                # Channel-Dimension entfernen [1, X, Y, Z] -> [X, Y, Z]
                patch_pred = prediction_batch[i].squeeze(0)
                
                # 1. DENORMALISIERUNG: Zurück in den Original-Wertebereich bringen
                patch_pred_denorm = (patch_pred * std_vol) + mean_vol
                    
                # 2. AKKUMULATION: Patch-Werte im Ziel-Volume aufaddieren
                predictions[volume_idx][x_start:x_end, y_start:y_end, z_start:z_end, timestep] += patch_pred_denorm
                
                # 3. COUNTER ERHÖHEN: Vermerken, dass diese Voxel abgedeckt wurden
                count_volumes[volume_idx][x_start:x_end, y_start:y_end, z_start:z_end, timestep] += 1.0



    # 4. MITTELWERTBILDUNG (DURCHSCHNITT): Überlappende Regionen glätten
    print("Berechne finalen Durchschnitt für überlappende Regionen...")
    for idx in range(len(predictions)):
        # Vermeide Division durch 0 (falls Voxel theoretisch gar nicht getroffen wurden)
        mask = count_volumes[idx] > 0
        predictions[idx][mask] /= count_volumes[idx][mask]
        
        # Optional: Falls Voxel gar nicht getroffen wurden (sollte bei deiner Grid-Logik nicht passieren)
        # kannst du sie hier auf 0 setzen oder maskieren.

    print("Rekonstruktion erfolgreich abgeschlossen!")
    return predictions



predictions = inference_and_reconstruction(model, gt, val_dataloader, device)
print("Prediction Volumes shape:")
print([vol.shape for vol in predictions])


utils.show_ram_usage()


# ### Visualisierung der Patches inkl. der Predictions

# In[ ]:


# Einzelne Testvisualisierung !!
# Richtige Visualisierungen alle nebeneinander & Difference Maps müssen wir weiter unten noch machen !

# Hole erstes Bild und Visualisiere
first_coords = coords[0]
Volume_Idx, Timestep, X_Start, X_End, Y_Start, Y_End, Z_Start, Z_End = first_coords

volume_pred = predictions[Volume_Idx]

print("Volume Pred shape:", volume_pred.shape)

image_pred = volume_pred[X_Start:X_End, Y_Start:Y_End, 0, Timestep]

plt.figure(figsize=(20,5))


plt.subplot(1,3,1)
plt.imshow(image_gt)
plt.title("GT Patch")

plt.subplot(1,3,2)
plt.imshow(image_dg)
plt.title("DG Patch")


plt.subplot(1,3,3)
plt.imshow(image_pred)
plt.title("Pred Patch")
plt.show()




# ### Non-Local-Means filter (NLM)

# In[ ]:


import time
import numpy as np
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma


def apply_nlm_to_predictions(predictions):
    """
    Wendet Non-Local Means auf alle rekonstruierten Prediction-Volumes an.

    Args:
        predictions: Liste von rekonstruierten 4D Volumes
                     Shape: (X, Y, Z, T)

    Returns:
        nlm_predictions: Liste gefilterter 4D Volumes
    """

    nlm_predictions = []

    total_start = time.time()

    for volume_idx, volume in enumerate(predictions):

        print(f"\n{'='*60}")
        print(f"Starte Volume {volume_idx + 1}/{len(predictions)}")
        print(f"Shape: {tuple(volume.shape)}")
        print(f"{'='*60}")

        volume_start = time.time()

        # Tensor -> NumPy nur EINMAL
        if hasattr(volume, "numpy"):
            volume_np = volume.numpy()
        else:
            volume_np = volume

        filtered_volume = np.zeros_like(volume_np, dtype=np.float32)

        timesteps = volume_np.shape[3]

        for t in tqdm(range(timesteps),
                      desc=f"Volume {volume_idx + 1}",
                      unit="Bild"):


            image_start = time.time()

            img_3d = volume_np[:, :, :, t]

            sigma_est = np.mean(
                estimate_sigma(
                    img_3d,
                    channel_axis=None
                )
            )

            filtered = denoise_nl_means(
                img_3d,
                h=2 * sigma_est,
                fast_mode=True,
                patch_size=3,
                patch_distance=2,
                channel_axis=None
            )

            filtered_volume[:, :, :, t] = filtered.astype(np.float32)

            elapsed = time.time() - image_start

        volume_time = time.time() - volume_start

        print(f"\n Volume {volume_idx + 1} abgeschlossen")
        print(f"Benötigte Zeit: {volume_time/60:.2f} Minuten")

        nlm_predictions.append(filtered_volume)

    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print("Non-Local Means vollständig abgeschlossen")
    print(f"Gesamtzeit: {total_time/60:.2f} Minuten")
    print(f"{'='*60}")

    return nlm_predictions


# In[ ]:


nlm_predictions = apply_nlm_to_predictions(predictions)

print("\nNLM abgeschlossen.")

if len(nlm_predictions) > 0:
    print("Shape des ersten Volumes:", nlm_predictions[0].shape)

utils.show_ram_usage()


# ### Show first comparison of whole predicted images and NLM filter image

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

volume_idx = 0
timestep = 0
z_slice = predictions[0].shape[2] // 2

pred_img = predictions[volume_idx][:, :, z_slice, timestep].numpy()
nlm_img = nlm_predictions[volume_idx][:, :, z_slice, timestep]


difference = pred_img - nlm_img  # Vorzeichenbehaftete Differenz

global_signal_max = np.max(pred_img)

# Verhindert Division durch 0, falls das Bild leer sein sollte
if global_signal_max == 0: 
    global_signal_max = 1.0

difference_percentage = ((pred_img - nlm_img) / global_signal_max) * 100
max_error = np.max(np.abs(difference_percentage))

vmin = min(pred_img.min(), nlm_img.min())
vmax = max(pred_img.max(), nlm_img.max())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# CNN Prediction
axes[0].imshow(
    pred_img.T,
    cmap="gray",
    origin="lower",
    vmin=vmin,
    vmax=vmax
)
axes[0].set_title("CNN Prediction (T=0)")
axes[0].axis("off")

# CNN + NLM
axes[1].imshow(
    nlm_img.T,
    cmap="gray",
    origin="lower",
    vmin=vmin,
    vmax=vmax
)
axes[1].set_title("CNN + NLM (T=0)")
axes[1].axis("off")

im = axes[2].imshow(
    difference_percentage.T,
    cmap="bwr",        # Blau = Negativ, Weiß = 0, Rot = Positiv
    origin="lower",
    vmin=-max_error,
    vmax=max_error,
)
axes[2].set_title("Signed Difference (%)")
axes[2].axis("off")

# Colorbar nur für die Differenz
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


# ## Evaluationsmetriken

# ### PSNR - Peak Signal Noise Ratio

# In[ ]:


import numpy as np
from skimage.metrics import peak_signal_noise_ratio

def compute_psnr_volumes(gt_volumes, pred_volumes):
    psnr_list = []

    for i, (gt, pred) in enumerate(zip(gt_volumes, pred_volumes)):

        # sicherstellen: numpy
        gt = np.asarray(gt, dtype=np.float32)
        pred = np.asarray(pred, dtype=np.float32)

        # Shape check
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch at volume {i}: {gt.shape} vs {pred.shape}")

        # PSNR über komplettes 4D Volume
        psnr = peak_signal_noise_ratio(
            gt,
            pred,
            data_range=gt.max() - gt.min()
        )

        psnr_list.append(psnr)
        print(f"Volume {i}: PSNR = {psnr:.4f} dB")

    mean_psnr = np.mean(psnr_list)

    # print("\n====================")
    # print(f"Mean PSNR: {mean_psnr:.4f} dB")
    # print("====================")

    return mean_psnr, psnr_list


# In[ ]:


#Ground Truth vs CNN Predictions only
mean_psnr_pred, psnr_pred_list = compute_psnr_volumes(gt, predictions)

#Ground Truth vs Degraded
mean_psnr_dg, psnr_dg_list = compute_psnr_volumes(gt, degraded)

#Ground Truth vs CNN Predictions + NLM Filter
#mean_psnr_nlm, psnr_nlm_list = compute_psnr_volumes(gt, nlm_predictions)


# Interpretation: Je höher der PSNR desto besser


# ### SSIM - Structural Similarity Index

# In[ ]:


from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_ssim_volumes(gt_volumes, pred_volumes):
    ssim_list = []

    for i, (gt, pred) in enumerate(zip(gt_volumes, pred_volumes)):

        gt = np.asarray(gt, dtype=np.float32)
        pred = np.asarray(pred, dtype=np.float32)

        # SSIM braucht 2D oder 3D → wir mitteln über Zeit
        t_steps = gt.shape[3]

        ssim_t = []

        for t in range(t_steps):
            s = ssim(
                gt[:, :, :, t],
                pred[:, :, :, t],
                data_range=gt.max() - gt.min()
            )
            ssim_t.append(s)

        mean_ssim = np.mean(ssim_t)
        ssim_list.append(mean_ssim)

        print(f"Volume {i}: SSIM = {mean_ssim:.4f}")

    # print("Mean SSIM:", np.mean(ssim_list))
    return np.mean(ssim_list), ssim_list


# In[ ]:


# Ground Truth vs CNN Predictions only
mean_ssim_pred, ssim_pred_list = compute_ssim_volumes(gt, predictions)

#Ground Truth vs Degraded
mean_ssim_dg, ssim_dg_list = compute_ssim_volumes(gt, degraded)

# Ground Truth vs CNN Predictions + NLM Filter
# mean_ssim_nlm, ssim_nlm_list = compute_ssim_volumes(gt, nlm_predictions)


# Interpretation SSIM:
# misst Strukturähnlichkeit statt nur Pixelfehler
# Wertebereich: SSIM ∈ [0, 1]; SSIM = 1: perfekte Übereinstimmung


# ### Pearson Correlation

# In[ ]:


from scipy.stats import pearsonr

def compute_pearson(gt_volumes, pred_volumes):
    corr_list = []

    for gt, pred in zip(gt_volumes, pred_volumes):

        gt = np.asarray(gt, dtype=np.float32).flatten()
        pred = np.asarray(pred, dtype=np.float32).flatten()

        corr, _ = pearsonr(gt, pred)

        corr_list.append(corr)
        print("Volume correlation:", corr)

    # print("Mean Pearson:", np.mean(corr_list))
    return np.mean(corr_list), corr_list


# In[ ]:


# Ground Truth vs CNN Predictions only
mean_pearson_pred, pearson_pred_list = compute_pearson(gt, predictions)

#Ground Truth vs Degraded
mean_pearson_dg, pearson_dg_list = compute_pearson(gt, degraded)

# Ground Truth vs CNN Predictions + NLM Filter
# mean_pearson_nlm, pearson_nlm_list = compute_pearson(gt, nlm_predictions)


# Interpretation Pearson Correlation: 
# misst funktionale Ähnlichkeit (sehr wichtig bei Gehirndaten)
# Wertebereich: r ∈ [-1, 1]; r = 1: perfekte lineare Übereinstimmung


# ## Visualisierungen

# ### Visualize the patches

# In[ ]:


# Choose val_dataloader because shuffle is set to false there. 
dg_batch, gt_batch, coords_of_patch, _, _ = next(iter(val_dataloader))
dg_batch = dg_batch.to(device)

batch_idx = 0
z_slice = 0
channel= 0

# Extrahiere die einzelnen Bilder aus den Batches
gt_batch = gt_batch[batch_idx, channel, :, :, z_slice]
pred_dg = model(dg_batch).detach().cpu()

pred_dg = pred_dg[batch_idx, channel, :, :, z_slice]
dg_batch_cpu = dg_batch.cpu()
dg_batch_cpu = dg_batch_cpu[batch_idx, channel, :, :, z_slice]

difference_map = gt_batch - pred_dg

# Erstelle eine Figur mit 4 Subplots in einer Reihe
plt.figure(figsize=(20, 6))

# Gesamtüberschrift für alle vier Abbildungen
plt.suptitle("Comparison of Patches: Ground Truth, Prediction, Degraded, and Difference Map", fontsize=16, y=1.02)

# Ground Truth
plt.subplot(1, 4, 1)
plt.imshow(gt_batch)
plt.title("Ground Truth (GT)")
plt.axis('off')

# Prediction
plt.subplot(1, 4, 2)
plt.imshow(pred_dg)
plt.title("Prediction")
plt.axis('off')

# Degraded
plt.subplot(1, 4, 3)
plt.imshow(dg_batch_cpu)
plt.title("Degraded")
plt.axis('off')

# Difference Map
plt.subplot(1, 4, 4)
plt.imshow(difference_map)
plt.title("Difference GT-Pred")
plt.axis('off')

plt.tight_layout()
plt.show()


# ### Visualize whole Image

# In[ ]:


# VISUALIZATION HYPERPARAMETERS
z_slice = gt[0].shape[2] // 2
t=0
volume_index = 0


# In[ ]:


plt.figure(figsize=(20, 6))
plt.suptitle("Comparison of entire image: GT, Degraded, Predictions and NLM applied to Predictions", fontsize=16, y=1.02)

# Ground Truth
plt.subplot(1, 4, 1)
image_gt = gt[volume_idx][:, :, z_slice, t]
plt.imshow(np.rot90(image_gt), cmap="gray")
plt.title("Ground Truth (GT)")
plt.axis('off')

# NLM Prediction
plt.subplot(1, 4, 2)
nlm_pred_slice = nlm_predictions[volume_idx][:, :, z_slice, t]
plt.imshow(np.rot90(nlm_pred_slice), cmap="gray")
plt.title("NLM Prediction")
plt.axis('off')

# Prediction
plt.subplot(1, 4, 3)
pred_slice = predictions[volume_idx][:, :, z_slice, t]
plt.imshow(np.rot90(pred_slice), cmap="gray")
plt.title("Prediction")
plt.axis('off')

# Degraded
plt.subplot(1, 4, 4)
degraded_slice = degraded[volume_idx][:, :, z_slice, t]
plt.imshow(np.rot90(degraded_slice), cmap="gray")
plt.title("Degraded")
plt.axis('off')

plt.tight_layout()
plt.show()


# ### Visualize Difference Maps

# In[ ]:


# Get images for difference calculations
pred_img = predictions[volume_idx][:, :, z_slice, t].numpy()
nlm_img = nlm_predictions[volume_idx][:, :, z_slice, t]
gt_img = gt[volume_idx][:, :, z_slice, t]
degraded_img = degraded[volume_idx][:, :, z_slice, t]

# Calculate global signal max for normalization
signal_max = max(np.max(pred_img), np.max(nlm_img), np.max(gt_img), np.max(degraded_img))
if signal_max == 0:
    signal_max = 1.0

# 1. Alle prozentualen Differenzen vorab berechnen
pred_diff = ((gt_img - pred_img) / signal_max) * 100
nlm_diff = ((gt_img - nlm_img) / signal_max) * 100
degraded_diff = ((gt_img - degraded_img) / signal_max) * 100
pred_deg_diff = ((degraded_img - pred_img) / signal_max) * 100
nlm_deg_diff = ((degraded_img - nlm_img) / signal_max) * 100

# 2. Das GLOBALE Maximum über ALLE Differenz-Maps hinweg finden
global_diff_max = max(np.max(np.abs(pred_diff)), np.max(np.abs(nlm_diff)), np.max(np.abs(degraded_diff)), np.max(np.abs(pred_deg_diff)), np.max(np.abs(nlm_deg_diff)))

# Create figure for difference maps
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
fig.suptitle("Difference Maps Comparison (%)", fontsize=16, y=1.02)
# Überall die identischen, symmetrischen Grenzen nutzen: vmin=-global_diff_max, vmax=global_diff_max


# GT vs Prediction
axes[0, 0].imshow(pred_diff.T, cmap="bwr", origin="lower", vmin=-global_diff_max, vmax=global_diff_max)
axes[0, 0].set_title("GT vs Prediction")
axes[0, 0].axis("off")

# GT vs NLM
axes[0, 1].imshow(nlm_diff.T, cmap="bwr", origin="lower", vmin=-global_diff_max, vmax=global_diff_max)
axes[0, 1].set_title("GT vs NLM")
axes[0, 1].axis("off")

# GT vs Degraded
axes[0, 2].imshow(degraded_diff.T, cmap="bwr", origin="lower", vmin=-global_diff_max, vmax=global_diff_max)
axes[0, 2].set_title("GT vs Degraded")
axes[0, 2].axis("off")

# Degraded vs Prediction
axes[1, 0].imshow(pred_deg_diff.T, cmap="bwr", origin="lower", vmin=-global_diff_max, vmax=global_diff_max)
axes[1, 0].set_title("Degraded vs Prediction")
axes[1, 0].axis("off")

# Degraded vs NLM
# Hier fangen wir das "im" Objekt ab, da es stellvertretend für ALLE Plots die richtige Skala hält
im = axes[1, 1].imshow(nlm_deg_diff.T, cmap="bwr", origin="lower", vmin=-global_diff_max, vmax=global_diff_max)
axes[1, 1].set_title("Degraded vs NLM")
axes[1, 1].axis("off")


# 3. Leeren Subplot unten rechts für die Colorbar nutzen
axes[1, 2].axis("off") # Achsenbeschriftung vom leeren Plot entfernen
# Colorbar direkt in das leere Feld setzen
fig.colorbar(im, ax=axes[1, 2], fraction=0.5, pad=0.5)

plt.tight_layout()
plt.show()


# ### Visualize k-Space

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Get images for difference calculations
pred_img = predictions[volume_idx][:, :, z_slice, t].numpy()
nlm_img = nlm_predictions[volume_idx][:, :, z_slice, t]
gt_img = gt[volume_idx][:, :, z_slice, t]
degraded_img = degraded[volume_idx][:, :, z_slice, t]

# --- HILFSFUNKTION FÜR K-RAUM ---
def to_log_kspace(img):
    # 2D Fourier-Transformation -> Shift -> Betrag (Magnitude)
    k_space = np.fft.fftshift(np.fft.fft2(img))
    magnitude = np.abs(k_space)
    # Logarithmische Skalierung: log(1 + x)
    return np.log(1 + magnitude)
    
# 2. k-Räume berechnen
k_gt = to_log_kspace(gt_img)
k_degraded = to_log_kspace(degraded_img)
k_pred = to_log_kspace(pred_img)
k_nlm = to_log_kspace(nlm_img)

# 3. k-Raum Differenzen berechnen (relativ zu Ground Truth)
diff_degraded = k_gt - k_degraded
diff_pred = k_gt - k_pred
diff_nlm = k_gt - k_nlm

# 4. Globale Skalierungsmaxima für absolute Vergleichbarkeit ermitteln
# Zeile 1: Absolute k-Raum Helligkeit (Graustufen)
vmax_k = max(k_gt.max(), k_degraded.max(), k_pred.max(), k_nlm.max())
vmin_k = min(k_gt.min(), k_degraded.min(), k_pred.min(), k_nlm.min())

# Zeile 2: Symmetrisches Maximum für die k-Raum Differenzen (bwr)
global_diff_max = max(
    np.max(np.abs(diff_degraded)),
    np.max(np.abs(diff_pred)),
    np.max(np.abs(diff_nlm))
)

# --- PLOTTING ---
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle("k-Space Analysis & Comparison (Log-Scaled)", fontsize=18, y=0.98)

# ZEILE 1: Die k-Räume (Graustufen)
titles_k = ["GT k-space", "Degraded k-space", "Prediction k-space", "NLM k-space"]
data_k = [k_gt, k_degraded, k_pred, k_nlm]

for i in range(4):
    im_k = axes[0, i].imshow(data_k[i].T, cmap="gray", origin="lower", vmin=vmin_k, vmax=vmax_k)
    axes[0, i].set_title(titles_k[i], fontsize=12)
    axes[0, i].axis("off")

# Colorbar für die k-Räume ganz rechts in Zeile 1 anhängen
fig.colorbar(im_k, ax=axes[0, 3], fraction=0.046, pad=0.04, label="Log Intensity")


# ZEILE 2: Die Differenzen (bwr - Blue-White-Red)
# Das erste Feld [1, 0] bleibt leer bzw. wird ausgeblendet, da GT vs GT keinen Sinn macht
axes[1, 0].axis("off") 

titles_diff = ["GT vs Degraded", "GT vs Prediction", "GT vs NLM"]
data_diff = [diff_degraded, diff_pred, diff_nlm]

for i in range(3):
    col_idx = i + 1 # Startet ab Spalte 1
    im_diff = axes[1, col_idx].imshow(
        data_diff[i].T, 
        cmap="bwr", 
        origin="lower", 
        vmin=-global_diff_max, 
        vmax=global_diff_max
    )
    axes[1, col_idx].set_title(f"Diff: {titles_diff[i]}", fontsize=12)
    axes[1, col_idx].axis("off")

# Colorbar für die Differenzen ganz rechts in Zeile 2 anhängen
fig.colorbar(im_diff, ax=axes[1, 3], fraction=0.046, pad=0.04, label="Difference Value")

plt.tight_layout()
plt.show()


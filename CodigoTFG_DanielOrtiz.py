# -*- coding: utf-8 -*-
"""
TFG — Clasificación automática de lesiones cutáneas mediante CNN
Dataset: ISIC 2018 Challenge — Task 3 (HAM10000)
Autor: Daniel Ortiz Requena
Universidad de Málaga — Ingeniería de la Salud
"""


# =================== INSTALACIÓN ===================
# Ejecutar UNA VEZ antes del pipeline y luego Runtime → Restart runtime:
#   !pip install albumentations -q
# Si aparece "torch._utils not found": Runtime → Factory reset runtime,
# luego instalar albumentations y ejecutar de nuevo.

# =================== IMPORTS ===================
import os, glob, shutil, random, warnings, time, json, sys, traceback
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import (
    resnet50,  ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
)
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, balanced_accuracy_score,
    precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize
from matplotlib.backends.backend_pdf import PdfPages
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================================================================
#  1. REPRODUCIBILIDAD Y DISPOSITIVO
# =============================================================================
# La semilla se fija en los cuatro sitios porque PyTorch, NumPy y Python
# tienen generadores de números aleatorios independientes. Si se fija solo
# en uno, los otros siguen siendo no deterministas.
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
torch.cuda.manual_seed_all(RNG_SEED)
# Sin estas dos líneas la GPU puede dar resultados distintos entre
# ejecuciones aunque la semilla esté fija, porque CuDNN elige kernels
# según disponibilidad en tiempo de ejecución.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✓ Dispositivo:", device)


# =============================================================================
#  2. CONFIGURACIÓN GLOBAL
# =============================================================================
# Las 7 clases del dataset en un orden fijo. Importante no cambiarlo
# porque cada posición corresponde a un índice numérico que usan los
# modelos — si se cambia, los checkpoints guardados dejan de ser válidos.
CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

CLASS_NAMES_FULL = {
    'MEL':   'Melanoma',
    'NV':    'Melanocytic Nevus',
    'BCC':   'Basal Cell Carcinoma',
    'AKIEC': 'Actinic Keratoses / Intraepithelial Carcinoma',
    'BKL':   'Benign Keratosis-like Lesions',
    'DF':    'Dermatofibroma',
    'VASC':  'Vascular Lesions',
}

# Resolución de entrada según el preentrenamiento de cada modelo.
# EfficientNet-B3 fue preentrenado con 300×300 — usar resolución menor
# penaliza su rendimiento al no aprovechar su escala de diseño.
IMG_SIZE = {
    'resnet50':        224,
    'densenet121':     224,
    'efficientnet_b3': 300,
}
IMG_SIZE_DEFAULT = 224

# Batch size 32 para ResNet y DenseNet: con BS=16 la validación oscilaba
# demasiado y los modelos paraban antes de tiempo por el ruido en la métrica.
# EfficientNet usa 8 porque sus imágenes de 300×300 ocupan más memoria GPU.
BATCH_SIZE = {
    'resnet50':        32,
    'densenet121':     32,
    'efficientnet_b3': 8,
}

# Split estratificado 60/20/20. El test set queda aislado hasta la
# evaluación final — no interviene en ninguna decisión de entrenamiento.
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2
TEST_RATIO  = 0.2
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

USE_PERCENT = 1.0
MAX_EPOCHS  = 35

# Patience de 10 epochs: valores menores (6-8) provocaban paradas prematuras
# en Kaggle por picos de validación. Con 10 los modelos convergen bien
# en ambos entornos (Colab y Kaggle).
ES_PATIENCE = {
    'resnet50':        10,
    'densenet121':     10,
    'efficientnet_b3': 10,
}
# Mejora mínima para considerar progreso real. DenseNet tiene delta más
# pequeño porque su curva de validación es más suave y los avances son
# incrementales en lugar de saltos bruscos.
ES_MIN_DELTA = {
    'resnet50':        0.001,
    'densenet121':     0.0005,
    'efficientnet_b3': 0.001,
}
# Peso del BACC en la puntuación mixta del early stopping.
# EfficientNet tiene un peso más alto (0.92) porque su val loss era
# inestable y darle demasiado peso a la pérdida provocaba paradas
# prematuras incluso cuando el BACC seguía mejorando.
ES_BACC_WEIGHTS = {
    'resnet50':        0.85,
    'densenet121':     0.85,
    'efficientnet_b3': 0.92,
}
# LR conservadores para fine-tuning sobre ImageNet.
# DenseNet y EfficientNet son más sensibles a LR altos —
# con valores mayores perdían los pesos preentrenados en las primeras epochs.
LR = {
    'resnet50':        1e-4,
    'densenet121':     5e-5,
    'efficientnet_b3': 5e-5,
}
# Regularización L2 proporcional al LR para mantener coherencia.
WEIGHT_DECAY = {
    'resnet50':        1e-4,
    'densenet121':     5e-5,
    'efficientnet_b3': 5e-5,
}
# El scheduler reduce el LR a la mitad si no mejora en N epochs.
# La patience es menor que la del early stopping (4-5 vs 10) para dar
# una segunda oportunidad con paso más pequeño antes de parar definitivamente.
SCHED_PATIENCE = {
    'resnet50':        4,
    'densenet121':     5,
    'efficientnet_b3': 4,
}

# Rondas de TTA: con 5 ya se observa mejora sobre argmax;
# con 10 mejora algo más con un coste de aproximadamente 2 minutos extra.
TTA_ROUNDS  = 10
MIN_SAMPLES = 10  # mínimo de imágenes por clase para incluirla en análisis

# Umbrales de ratio para asignar nivel de augmentation a cada clase:
#   ratio < 2 → nivel 0  |  2-6 → nivel 1  |  6-15 → nivel 2  |  >15 → nivel 3
AUG_THR_LIGHT  = 2.0
AUG_THR_MEDIUM = 6.0
AUG_THR_HEAVY  = 15.0

# =============================================================================
#  3. CONFIGURACION DE ENTORNO
#  Detecta automaticamente Colab / Kaggle / local.
#  Solo tocar las variables de la seccion "LO UNICO QUE DEBES TOCAR".
#
#  COLAB  : SOURCE/CSV en Drive, PREP en RAM local, OUTPUTS en Drive.
#  KAGGLE : SOURCE/CSV en /kaggle/input, PREP y OUTPUTS en /kaggle/working.
#  LOCAL  : todo bajo LOCAL_BASE_DIR.
#
#  NUM_WORKERS: 0 en Colab (evita torch._utils error), 2 en Kaggle/local.
#  BATCH_SIZE B3: 8 en todos (seguro en T4/P100 a 300px).
#
#  LO UNICO QUE DEBES TOCAR
# =============================================================================
KAGGLE_DATASET_SLUG = "danielortizrequena/isic2018-task3"
COLAB_DRIVE_FOLDER  = "ISIC2018_Task3"
LOCAL_BASE_DIR      = os.path.expanduser("~/ISIC2018_Task3")


def _detect_env():
    # Kaggle se detecta por sus variables de entorno propias, no por la
    # presencia de google.colab, que puede ser importable en Kaggle por
    # su capa de compatibilidad. Las variables KAGGLE_* son el único
    # indicador inequívoco del entorno Kaggle.
    if os.environ.get('KAGGLE_DATA_PROXY_TOKEN') or os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    if os.path.isdir("/kaggle/input"):
        return "kaggle"
    return "local"

_ENV = _detect_env()

if _ENV == "kaggle":
    # Kaggle monta datasets en /kaggle/input/<nombre>/ pero el nombre exacto
    # varía según la versión de la API y cómo se añade el dataset al notebook.
    # Un único glob sobre el CSV de referencia descubre la ruta real sin
    # necesidad de conocer el prefijo de montaje.
    _csv_ref  = "ISIC2018_Task3_Training_GroundTruth.csv"
    _csv_hits = glob.glob(f"/kaggle/input/**/{_csv_ref}", recursive=True)
    if _csv_hits:
        CSV_DIR    = os.path.dirname(_csv_hits[0])
        SOURCE_DIR = os.path.join(os.path.dirname(CSV_DIR), "ISIC2018")
    else:
        # Fallback si el glob no encuentra nada (dataset no añadido al notebook)
        _ds_name   = KAGGLE_DATASET_SLUG.split("/")[-1]
        SOURCE_DIR = f"/kaggle/input/{_ds_name}/ISIC2018_Task3/ISIC2018"
        CSV_DIR    = f"/kaggle/input/{_ds_name}/ISIC2018_Task3/Groundtruth"
    PREP_DIR     = "/kaggle/working/isic2018_task3_prepared"
    OUTPUTS_ROOT = "/kaggle/working/OUTPUTS"
    NUM_WORKERS  = 2
    BATCH_SIZE['efficientnet_b3'] = 8

elif _ENV == "colab":
    try:
        from google.colab import drive as _gdrive
        _gdrive.mount('/content/drive')
        print("Google Drive montado")
    except Exception as _e:
        print(f"No se pudo montar Drive: {_e}")
    SOURCE_DIR   = f"/content/drive/MyDrive/{COLAB_DRIVE_FOLDER}/ISIC2018"
    CSV_DIR      = f"/content/drive/MyDrive/{COLAB_DRIVE_FOLDER}/Groundtruth"
    PREP_DIR     = "/content/data/isic2018_task3_prepared"
    OUTPUTS_ROOT = f"/content/drive/MyDrive/{COLAB_DRIVE_FOLDER}/OUTPUTS"
    NUM_WORKERS  = 0
    BATCH_SIZE['efficientnet_b3'] = 8

else:
    SOURCE_DIR   = os.path.join(LOCAL_BASE_DIR, "ISIC2018")
    CSV_DIR      = os.path.join(LOCAL_BASE_DIR, "Groundtruth")
    PREP_DIR     = os.path.join(LOCAL_BASE_DIR, "prepared")
    OUTPUTS_ROOT = os.path.join(LOCAL_BASE_DIR, "OUTPUTS")
    NUM_WORKERS  = 2
    BATCH_SIZE['efficientnet_b3'] = 8

print(f"\n{'='*60}")
print(f"  Entorno           : {_ENV.upper()}")
print(f"  SOURCE_DIR        : {SOURCE_DIR}")
print(f"  CSV_DIR           : {CSV_DIR}")
print(f"  PREP_DIR          : {PREP_DIR}")
print(f"  OUTPUTS_ROOT      : {OUTPUTS_ROOT}")
print(f"  NUM_WORKERS       : {NUM_WORKERS}")
print(f"  BS EfficientNet-B3: {BATCH_SIZE['efficientnet_b3']}")
print(f"{'='*60}")
for _label, _d in [("SOURCE_DIR", SOURCE_DIR), ("CSV_DIR", CSV_DIR)]:
    _status = "OK" if os.path.isdir(_d) else "NO ENCONTRADO - revisa la ruta arriba"
    print(f"  {_status}  {_label}: {_d}")

# =============================================================================
#  MODO DE EJECUCIÓN — solo uno activo a la vez
#  Si LOAD_TTA_FROM_DIR está activo, los otros se ignoran.
# =============================================================================

# Modo 1: solo umbrales — carga arrays de ejecución anterior, salta entrenamiento.
# Útil para recalibrar umbrales clínicos sin reentrenar los modelos.
# Cambiar a la ruta completa de la carpeta que contiene los .npy guardados.
# Ej: "/content/drive/MyDrive/ISIC2018_Task3/OUTPUTS/2026-04-20_17-04-28"
LOAD_TTA_FROM_DIR = False

# Modo 2: reanudar — carga checkpoints existentes en RESUME_DIR,
# entrena los modelos que no tengan checkpoint. Crea carpeta nueva _r.
# Ej: "/content/drive/MyDrive/ISIC2018_Task3/OUTPUTS/2026-04-20_17-04-28"
RESUME_FROM_CHECKPOINTS = False
RESUME_DIR = ""

# Modo 2b: FORCE_RETRAIN — lista de modelos que se reentrenan AUNQUE
# exista su checkpoint en la carpeta de reanudación. El resto se carga.
#
# Cuándo usarlo: cuando un modelo convergió demasiado pronto o tarde por
# variabilidad de sesión (Colab/Kaggle asigna GPUs distintas con diferente
# rendimiento) y se quiere corregir sin perder los checkpoints de los modelos
# que sí funcionaron bien.
#
# Ejemplo de uso:
#   RESUME_FROM_CHECKPOINTS = True
#   RESUME_DIR = "<ruta carpeta anterior>"
#   FORCE_RETRAIN = ['efficientnet_b3']   # solo reentrenar este
#
# El nuevo checkpoint sobreescribe el copiado desde RESUME_DIR —
# los archivos originales no se tocan.
FORCE_RETRAIN = []   # ← añadir nombres para forzar reentrenamiento
                     #   ej: ['efficientnet_b3']  o  ['resnet50', 'densenet121']

# =============================================================================
#  MEJORAS CLÍNICAS
# =============================================================================
# gamma=0 en EfficientNet equivale a CrossEntropy estándar. Con gamma=1
# tendía a sobreajustar en las primeras epochs para este modelo concreto,
# por lo que se mantiene gamma=0 solo para él.
FOCAL_GAMMA = {
    'resnet50':        1.0,
    'densenet121':     1.0,
    'efficientnet_b3': 0.0,
}

# MEL le correspondería nivel 1 de augmentation por su ratio de desbalanceo,
# pero se sube manualmente a nivel 2 para mejorar su sensibilidad.
# En experimentos anteriores sin este boost la sensibilidad de MEL era ~0.65.
CLINICAL_AUG_BOOST = {
    'MEL': 2,
}

# Configuración de umbrales clínicos para MEL y AKIEC.
#
# DISEÑO DE LA ESTRATEGIA DE CALIBRACIÓN:
#
# Se probaron dos enfoques para calibrar el umbral de MEL:
#
# 1. F-beta (β=2) con specificity_floor=0.70:
#    Encontraba umbrales muy agresivos (theta≈0.15) que mejoraban la
#    sensibilidad de MEL pero destruían la de NV (-0.131). NV es el 67%
#    del dataset y clasificarlo mal tiene consecuencias clínicas reales
#    (falso positivo = biopsia evitable mal descartada). El coste en BACC
#    global llegaba a -0.048, 10 veces peor que la alternativa.
#
# 2. argmax-θ con specificity_floor=0.85:
#    Busca el theta más alto que cumple sens≥0.85 y spec≥0.85 en val.
#    Con buenos modelos se encuentra theta≈0.31-0.34. El daño colateral
#    en NV y el BACC global es mínimo (delta≈-0.004).
#
# Conclusión: argmax-θ con floor=0.85 alcanza el objetivo clínico
# (sens MEL≥0.85) sin sacrificar la clase mayoritaria. No hay justificación
# para empeorar NV si el objetivo ya se cubre con la estrategia conservadora.
#
# AKIEC usa floor=0.70 porque tiene solo 75 muestras en validación —
# exigir spec≥0.85 con tan pocas muestras deja sin candidatos válidos.
CLINICAL_THRESHOLDS = {
    'MEL':   {'sensitivity_target': 0.85, 'specificity_floor': 0.85,
              'use_fbeta': False, 'beta': 1.0},
    'AKIEC': {'sensitivity_target': 0.75, 'specificity_floor': 0.70,
              'use_fbeta': False, 'beta': 1.0},
}
THRESHOLD_PRIORITY = ['MEL', 'AKIEC']

MEL_SENSITIVITY_TARGET = CLINICAL_THRESHOLDS['MEL']['sensitivity_target']
MEL_SPECIFICITY_FLOOR  = CLINICAL_THRESHOLDS['MEL']['specificity_floor']


# =============================================================================
#  GLOBAL PARA MANEJO DE ERRORES + TEE
# =============================================================================
_current_out_dir = None   # Actualizado al inicio de main() para el bloque __main__


class Tee:
    """
    Redirige sys.stdout a pantalla Y a un archivo de log simultáneamente.
    Así el output de Colab queda guardado en Drive aunque se cierre la sesión.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, 'w', encoding='utf-8', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()


# =============================================================================
#  4. UTILIDADES
# =============================================================================
def log(msg: str):
    """Imprime con flush inmediato — con Tee activo va a pantalla y al .txt."""
    print(msg, flush=True)


def load_groundtruth_csv(csv_path):
    """Carga un único CSV de ground truth de ISIC y devuelve {image_id: clase}."""
    df      = pd.read_csv(csv_path)
    key_col = 'image' if 'image' in df.columns else df.columns[0]
    cls_cols = [c for c in df.columns if c.upper() in CLASSES]
    result = {}
    for _, row in df.iterrows():
        img_id = str(row[key_col])
        for col in cls_cols:
            val = row[col]
            if val == 1 or str(val).strip() == "1":
                result[img_id] = col.upper()
                break
    return result


def merge_groundtruth_csvs(csv_dir):
    """
    Fusiona los tres CSVs del challenge ISIC 2018 Task 3 en un único diccionario
    {image_id: clase}. El challenge original divide los datos en train/val/test
    propios; los fusionamos para poder hacer nuestro propio split estratificado
    con control total sobre las proporciones y la semilla.
    """
    # ISIC 2018 distribuye las etiquetas en tres CSVs separados (train/val/test
    # del challenge original). Los fusiono en un único diccionario
    # {id_imagen: clase} para poder hacer mi propio split estratificado.
    paths = {
        "train": os.path.join(csv_dir, "ISIC2018_Task3_Training_GroundTruth.csv"),
        "val":   os.path.join(csv_dir, "ISIC2018_Task3_Validation_GroundTruth.csv"),
        "test":  os.path.join(csv_dir, "ISIC2018_Task3_Test_GroundTruth.csv"),
    }
    for s, p in paths.items():
        assert os.path.exists(p), f"CSV no encontrado ({s}): {p}"
    label_map, origin, conflicts = {}, {}, []
    for s, p in paths.items():
        for img_id, cls in load_groundtruth_csv(p).items():
            if img_id in label_map and label_map[img_id] != cls:
                conflicts.append((img_id, label_map[img_id], cls, origin[img_id], s))
            else:
                label_map[img_id] = cls
                origin[img_id]    = s
    if conflicts:
        raise RuntimeError(f"Conflictos de etiqueta: {conflicts[:3]}")
    log(f"✓ Ground truth cargado: {len(label_map)} imágenes")
    return label_map


def index_images(folder):
    """
    Construye {image_id: ruta_archivo} para todas las imágenes bajo una carpeta.
    Los IDs se ordenan lexicográficamente para garantizar reproducibilidad
    independientemente del orden del sistema de archivos.
    """
    per_id = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for path in glob.glob(os.path.join(folder, "**", ext), recursive=True):
            img_id = os.path.splitext(os.path.basename(path))[0]
            if img_id not in per_id:
                per_id[img_id] = path
    # Sin este sorted(), cada sesión de Colab devuelve los archivos en
    # distinto orden según el sistema de ficheros, y el split cambia
    # aunque la semilla sea la misma. El orden lexicográfico garantiza
    # que el split es reproducible entre sesiones.
    return dict(sorted(per_id.items()))


# =============================================================================
#  5. SPLIT ESTRATIFICADO
# =============================================================================
def split_dataset(label_map, id2path, train_ratio=0.6, val_ratio=0.2,
                  use_percent=1.0, save_path=None):
    """
    Split estratificado por clase con orden completamente determinista.

    Hay dos fuentes de no-determinismo que se corrigen explícitamente:
    1. index_images devuelve ids ordenados, pero además se ordenan los ids
       de cada clase ANTES del shuffle para que random.seed() produzca
       siempre el mismo resultado independientemente del orden de glob.
    2. Las clases se iteran en orden alfabético (sorted), no en orden de
       inserción del dict, que dependía del orden de glob en cada sesión.

    Con semilla fija + orden fijo el split es reproducible entre sesiones
    de Colab, Kaggle y ejecuciones locales.

    save_path: si se especifica, guarda {image_id: split} como JSON
    para que futuras reanudaciones usen exactamente el mismo split.
    """
    assert train_ratio + val_ratio < 1.0
    log(f"ℹ Split estratificado "
        f"({int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)} garantizado por clase)")
    per_cls = defaultdict(list)
    for iid in id2path:
        if iid in label_map:
            per_cls[label_map[iid]].append(iid)
    random.seed(RNG_SEED)
    train_ids, val_ids, test_ids = set(), set(), set()

    for cls in sorted(per_cls.keys()):
        ids = sorted(per_cls[cls])
        random.shuffle(ids)
        if use_percent < 1.0:
            ids = ids[:max(1, int(len(ids) * use_percent))]
        n = len(ids); nt = int(n * train_ratio); nv = int(n * val_ratio)
        train_ids.update(ids[:nt])
        val_ids.update(ids[nt:nt + nv])
        test_ids.update(ids[nt + nv:])

    prep = {s: defaultdict(list) for s in ('train', 'val', 'test')}
    for iid, path in id2path.items():
        cls = label_map.get(iid)
        if cls not in CLASSES: continue
        if iid in train_ids:   prep['train'][cls].append(path)
        elif iid in val_ids:   prep['val'][cls].append(path)
        elif iid in test_ids:  prep['test'][cls].append(path)

    tt = sum(len(v) for v in prep['train'].values())
    tv = sum(len(v) for v in prep['val'].values())
    ts = sum(len(v) for v in prep['test'].values())
    log(f"✓ Split completado — Train: {tt} | Val: {tv} | Test: {ts}")

    if save_path:
        assignment = {}
        for iid in train_ids: assignment[iid] = 'train'
        for iid in val_ids:   assignment[iid] = 'val'
        for iid in test_ids:  assignment[iid] = 'test'
        with open(save_path, 'w') as fp:
            json.dump(assignment, fp)
        log(f"✓ Split guardado: {os.path.basename(save_path)}")

    return dict(prep['train']), dict(prep['val']), dict(prep['test'])


def load_split_from_file(split_path, id2path, label_map):
    """
    Reconstruye prep_train/val/test desde split_assignment.json guardado.
    Garantiza que la reanudación usa EXACTAMENTE el mismo split que el run
    original, evitando cualquier riesgo de data leakage por cambios de orden.
    """
    with open(split_path) as f:
        assignment = json.load(f)
    prep = {s: defaultdict(list) for s in ('train', 'val', 'test')}
    missing = 0
    for iid, path in id2path.items():
        cls = label_map.get(iid)
        if cls not in CLASSES: continue
        sp = assignment.get(iid)
        if sp in prep:
            prep[sp][cls].append(path)
        else:
            missing += 1
    if missing:
        log(f"  ⚠ {missing} imágenes no encontradas en split_assignment.json — omitidas")
    tt = sum(len(v) for v in prep['train'].values())
    tv = sum(len(v) for v in prep['val'].values())
    ts = sum(len(v) for v in prep['test'].values())
    log(f"✓ Split cargado desde archivo — Train: {tt} | Val: {tv} | Test: {ts}")
    return dict(prep['train']), dict(prep['val']), dict(prep['test'])


# =============================================================================
#  6. DATASET CON AUGMENTATION GRADUADA
# =============================================================================
class SkinDataset(Dataset):
    """
    Dataset con augmentation graduada por clase.

    Los transforms se guardan como atributos de la instancia, no como variables
    globales. El motivo es que al entrenar tres modelos seguidos con resoluciones
    distintas (224px y 300px), si fueran globales el segundo modelo sobreescribiría
    los del primero y al evaluar usaría la resolución incorrecta, produciendo
    métricas erróneas sin ningún error visible en el código.
    """
    def __init__(self, root_dir, classes, is_train=False, transforms=None):
        self.samples  = []
        self.classes  = classes
        self.cls2idx  = {c: i for i, c in enumerate(classes)}
        self.is_train = is_train

        _t = transforms if transforms is not None else build_transforms(IMG_SIZE_DEFAULT)
        self._aug_transforms = _t['aug']
        self._val_transform  = _t['val']
        self._tta_transform  = _t['tta']

        for cls in classes:
            cls_dir = Path(root_dir) / cls
            if cls_dir.exists():
                for p in sorted(cls_dir.iterdir()):
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        self.samples.append((str(p), cls))

        if is_train:
            counts    = Counter(cls for _, cls in self.samples)
            max_count = max(counts.values())
            self.class_aug_level = {}
            for cls, n in counts.items():
                r = max_count / n
                if r < AUG_THR_LIGHT:    level = 0
                elif r < AUG_THR_MEDIUM: level = 1
                elif r < AUG_THR_HEAVY:  level = 2
                else:                    level = 3
                self.class_aug_level[cls] = level

            # El boost clínico sube el nivel de augmentation para clases
            # cuya importancia clínica supera lo que el ratio asigna
            # automáticamente. En este caso solo MEL recibe boost.
            for cls, min_level in CLINICAL_AUG_BOOST.items():
                if cls in self.class_aug_level:
                    self.class_aug_level[cls] = max(self.class_aug_level[cls], min_level)

            log(f"\n  Augmentation graduada (clase mayoritaria: {max_count} imgs):")
            for cls in sorted(counts, key=lambda c: counts[c], reverse=True):
                r     = max_count / counts[cls]
                level = self.class_aug_level[cls]
                base  = (0 if r < AUG_THR_LIGHT else
                         1 if r < AUG_THR_MEDIUM else
                         2 if r < AUG_THR_HEAVY else 3)
                boost = " ★BOOST CLÍNICO" if cls in CLINICAL_AUG_BOOST and level > base else ""
                log(f"    {cls:8s}  {counts[cls]:5d} imgs  ratio={r:5.1f}"
                    f"  — Nivel {level} ({AUG_LEVEL_NAMES[level]}){boost}")
        else:
            self.class_aug_level = {cls: 0 for cls in classes}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if getattr(self, '_tta_mode', False):
            img = self._tta_transform(image=img)['image']
        else:
            img = self._aug_transforms[self.class_aug_level.get(cls, 0)](image=img)['image']
        return img, self.cls2idx[cls]


# =============================================================================
#  7. TRANSFORMACIONES — 4 NIVELES GRADUADOS
# =============================================================================
_NORM = [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ToTensorV2()]


def build_transforms(img_size):
    """
    4 niveles de augmentation según el desbalanceo de cada clase:
      0 — solo resize+normalización (val y test siempre usan este)
      1 — giros y rotaciones suaves (clases con ratio 2-6)
      2 — añade deformaciones geométricas (clases con ratio 6-15)
      3 — añade ruido y desenfoque (clases con ratio >15, muy pocas muestras)

    Val y test usan siempre nivel 0 para que la evaluación sea reproducible.
    TTA usa solo giros y volteos geométricos, sin cambios de color, porque
    el calibrador de umbrales se ajustó con colores reales y alterarlos
    desajustaría los thresholds calculados sobre validación.
    """
    aug0 = A.Compose([A.Resize(img_size, img_size), *_NORM])
    aug1 = A.Compose([A.Resize(img_size, img_size),
                      A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                      A.Rotate(limit=30, p=0.6), *_NORM])
    aug2 = A.Compose([A.Resize(img_size, img_size),
                      A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                      A.Rotate(limit=90, p=0.7),
                      A.Affine(translate_percent={"x": 0.05, "y": 0.05},
                               scale=(0.9, 1.1), p=0.4), *_NORM])
    aug3 = A.Compose([A.Resize(img_size, img_size),
                      A.Affine(translate_percent={"x": 0.1, "y": 0.1},
                               scale=(0.9, 1.1), rotate=(-20, 20), p=0.5),
                      A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                      A.Rotate(limit=180, p=0.7),
                      A.GaussNoise(std_range=(0.02, 0.10), p=0.3),
                      A.MotionBlur(blur_limit=3, p=0.2),
                      A.MedianBlur(blur_limit=3, p=0.1),
                      A.ImageCompression(quality_range=(75, 100), p=0.3), *_NORM])
    val_t = aug0
    tta_t = A.Compose([A.Resize(img_size, img_size),
                       A.RandomRotate90(p=0.5), A.HorizontalFlip(p=0.5),
                       A.VerticalFlip(p=0.5), *_NORM])
    return {'aug': {0: aug0, 1: aug1, 2: aug2, 3: aug3}, 'val': val_t, 'tta': tta_t}


_transforms_224 = build_transforms(IMG_SIZE_DEFAULT)
AUG_TRANSFORMS  = _transforms_224['aug']
val_transform   = _transforms_224['val']
tta_transform   = _transforms_224['tta']

AUG_LEVEL_NAMES = {0: "sin augmentation", 1: "leve", 2: "media", 3: "agresiva"}


# =============================================================================
#  FOCAL LOSS CON LABEL SMOOTHING
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss con label smoothing aplicado solo en entrenamiento.

    El label smoothing suaviza las etiquetas duras (0/1) para evitar
    sobreconfianza en clases con pocas muestras, donde el modelo tendería
    a saturar las probabilidades.

    No se aplica en validación porque hace que la pérdida sea artificialmente
    alta, lo que confundía al scheduler de LR: bajaba el learning rate antes
    de tiempo al interpretar la pérdida elevada como estancamiento.
    Con CrossEntropyLoss estándar en validación, la pérdida es directamente
    interpretable y el scheduler actúa en el momento correcto.
    """
    def __init__(self, gamma=1.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.gamma = gamma; self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        C = logits.size(1)
        with torch.no_grad():
            smooth = torch.full_like(logits, self.label_smoothing / (C - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        log_probs     = F.log_softmax(logits, dim=1)
        ce_per_sample = -(smooth * log_probs).sum(dim=1)
        with torch.no_grad():
            p_t = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
            fw  = (1.0 - p_t) ** self.gamma
        loss = fw * ce_per_sample
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# =============================================================================
#  8. SAMPLER PONDERADO
# =============================================================================
def build_weighted_sampler(dataset):
    """
    Crea un WeightedRandomSampler que asigna mayor probabilidad de muestreo
    a las clases poco frecuentes. Contrarresta el desbalanceo del dataset
    (NV representa ~67% de las imágenes) sin modificar las etiquetas.

    Returns:
        WeightedRandomSampler con pesos inversamente proporcionales al tamaño de cada clase.
    """
    # Da más probabilidad de aparecer en cada batch a las imágenes de
    # clases poco frecuentes, compensando el fuerte desbalanceo del dataset
    # (NV tiene ~67% de los datos, VASC y DF menos del 2%).
    counts  = Counter(cls for _, cls in dataset.samples)
    weights = {cls: 1.0 / n for cls, n in counts.items()}
    sw      = torch.tensor([weights[cls] for _, cls in dataset.samples], dtype=torch.float)
    return WeightedRandomSampler(sw, len(sw), replacement=True)


# =============================================================================
#  9. MODELOS PREENTRENADOS
# =============================================================================
def build_model(name, num_classes):
    """
    Instancia un backbone preentrenado en ImageNet con la cabeza clasificadora
    reemplazada para num_classes salidas.

    Args:
        name:        'resnet50', 'densenet121' o 'efficientnet_b3'
        num_classes: número de clases de salida (7 en este proyecto)

    Returns:
        nn.Module listo para fine-tuning, movido al dispositivo global.
    """
    # En cada modelo solo se cambia la última capa clasificadora para que
    # produzca 7 clases en lugar de las 1000 de ImageNet. El resto de pesos
    # preentrenados se ajusta durante el fine-tuning completo.
    if name == 'resnet50':
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == 'densenet121':
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == 'efficientnet_b3':
        m = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Modelo desconocido: {name}")
    return m.to(device)


# =============================================================================
#  10. EARLY STOPPING
# =============================================================================
class EarlyStopping:
    """
    Para el entrenamiento si el modelo no mejora en N epochs.

    Usa una puntuación mixta: score = w·BACC + (1-w)·1/(1+loss).
    Esta combinación es más estable que mirar solo BACC porque la pérdida
    suaviza las oscilaciones puntuales de la métrica de validación, que
    en datasets desbalanceados puede variar bastante entre epochs seguidas.
    """
    def __init__(self, patience=8, min_delta=0.001, checkpoint_path=None, bacc_weight=0.8):
        self.patience = patience; self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path; self.bacc_weight = bacc_weight
        self.best_score = -np.inf; self.best_bacc = 0.0; self.best_loss = np.inf
        self.counter = 0; self.best_epoch = 0

    def _score(self, val_bacc, val_loss):
        return self.bacc_weight * val_bacc + (1 - self.bacc_weight) / (1 + val_loss)

    def step(self, val_bacc, val_loss, model, epoch):
        score = self._score(val_bacc, val_loss)
        if score > self.best_score + self.min_delta:
            self.best_score = score; self.best_bacc = val_bacc
            self.best_loss  = val_loss; self.counter = 0; self.best_epoch = epoch
            if self.checkpoint_path:
                torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
        return self.counter >= self.patience


class _CheckpointMeta:
    """Sustituye a EarlyStopping cuando se carga un modelo desde checkpoint."""
    def __init__(self, meta, bacc_weight):
        self.best_epoch  = meta['best_epoch']
        self.best_bacc   = meta['val_bacc']
        self.best_loss   = meta.get('val_loss', 0.0)
        self.bacc_weight = bacc_weight

    def _score(self, val_bacc, val_loss):
        return self.bacc_weight * val_bacc + (1 - self.bacc_weight) / (1 + val_loss)


# =============================================================================
#  11. BUCLE DE ENTRENAMIENTO
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    """
    Ejecuta una epoch completa de entrenamiento sobre el DataLoader dado.

    Incluye gradient clipping (max_norm=1.0) para evitar explosión de gradientes
    en clases minoritarias con pocos ejemplos en el batch.

    Returns:
        Tupla (avg_loss, accuracy, balanced_accuracy) sobre la epoch completa.
    """
    model.train()
    run_loss = cor = total = 0; pa, la = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x); loss = criterion(logits, y)
        loss.backward()
        # Gradient clipping: limita el gradiente para evitar saltos bruscos
        # cuando el modelo falla mucho en una clase poco frecuente y el
        # gradiente es muy grande.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        run_loss += loss.item() * x.size(0)
        p = logits.argmax(1); cor += (p == y).sum().item(); total += y.size(0)
        pa.extend(p.cpu().numpy()); la.extend(y.cpu().numpy())
    return run_loss / total, cor / total, balanced_accuracy_score(la, pa)


def evaluate(model, loader, criterion, classes_used):
    """
    Evaluación completa sobre un DataLoader sin actualizar pesos.

    Returns:
        Tupla (avg_loss, accuracy, balanced_accuracy, confusion_matrix,
               probs, labels) donde probs es un array (N, C) de probabilidades
               softmax y labels es la lista de etiquetas verdaderas.
    """
    model.eval()
    run_loss = cor = total = 0; pa, la, pra = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x); loss = criterion(logits, y)
            probs  = torch.softmax(logits, 1)
            run_loss += loss.item() * x.size(0)
            p = logits.argmax(1); cor += (p == y).sum().item(); total += y.size(0)
            pa.extend(p.cpu().numpy()); la.extend(y.cpu().numpy())
            pra.append(probs.cpu().numpy())
    pra = np.vstack(pra)
    return (run_loss / total, cor / total,
            balanced_accuracy_score(la, pa),
            confusion_matrix(la, pa), pra, la)


# =============================================================================
#  12. TEST-TIME AUGMENTATION (TTA)
# =============================================================================
def predict_with_tta(model, dataset, n_rounds=10, batch_size=16):
    """
    Realiza N predicciones con augmentation aleatoria y promedia las
    probabilidades. Esto reduce la varianza de la predicción y mejora
    el BACC especialmente en clases minoritarias, donde una sola predicción
    puede ser muy sensible a pequeñas variaciones de la imagen.

    _tta_mode se restaura en el bloque finally para no dejar el dataset
    en modo incorrecto si ocurre un error durante la inferencia.
    """
    model.eval(); sum_probs = None; all_labels = []
    dataset._tta_mode = True
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)
    try:
        for _ in range(n_rounds):
            rp, rl = [], []
            with torch.no_grad():
                for x, y in loader:
                    rp.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
                    rl.extend(y.numpy())
            rp = np.vstack(rp)
            if sum_probs is None: sum_probs = rp; all_labels = rl
            else: sum_probs += rp
    finally:
        dataset._tta_mode = False
    return sum_probs / n_rounds, np.array(all_labels)


# =============================================================================
#  13. GRAD-CAM
# =============================================================================
def compute_gradcam(model, image_tensor, target_class, model_name):
    """
    Calcula el mapa de calor Grad-CAM para una imagen y una clase objetivo.

    Usa la última capa convolucional de cada arquitectura — la que captura
    características semánticas de mayor nivel antes del pooling global.
    Sirve para verificar que el modelo atiende a la lesión y no a artefactos.

    Args:
        model:         modelo entrenado en modo eval
        image_tensor:  tensor (C, H, W) normalizado
        target_class:  índice de la clase a explicar
        model_name:    'resnet50', 'densenet121' o 'efficientnet_b3'

    Returns:
        Array (H', W') normalizado en [0, 1]. Valores altos = mayor activación.
    """
    # Mapa de calor que muestra qué zonas de la imagen activaron la predicción.
    # Sirve para verificar que el modelo atiende a la lesión y no a artefactos
    # del fondo (pelo, regla dermatoscópica, marcos). Se usa la última capa
    # convolucional de cada arquitectura, que captura las características
    # semánticas de mayor nivel.
    if model_name == 'resnet50':
        tl = model.layer4[-1].conv3
    elif model_name == 'densenet121':
        tl = model.features.denseblock4.denselayer16.conv2
    elif model_name == 'efficientnet_b3':
        tl = model.features[8][0]
    else:
        raise ValueError(f"Grad-CAM no impl.: {model_name}")

    grads, acts = [], []
    def _sg(g): grads.append(g.detach().cpu())
    def _fh(m, i, o): acts.append(o.detach().cpu()); o.register_hook(_sg)

    handle = tl.register_forward_hook(_fh); model.eval()
    out = model(image_tensor.unsqueeze(0).to(device))
    model.zero_grad(); out[0, target_class].backward(); handle.remove()

    g = grads[0].squeeze().numpy(); a = acts[0].squeeze().numpy()
    cam = np.maximum(np.einsum('c,chw->hw', g.mean(axis=(1, 2)), a), 0)
    mn, mx = cam.min(), cam.max()
    if mx - mn > 1e-8: cam = (cam - mn) / (mx - mn)
    return cam


def overlay_gradcam(img, cam):
    """
    Superpone el mapa de calor Grad-CAM sobre la imagen original.

    Args:
        img: array (H, W, 3) uint8 RGB
        cam: array (H', W') en [0, 1], se redimensiona automáticamente

    Returns:
        Array (H, W, 3) uint8 con la mezcla 60% imagen + 40% heatmap jet.
    """
    cam_r = np.array(Image.fromarray((cam * 255).astype(np.uint8))
                     .resize((img.shape[1], img.shape[0]), Image.BILINEAR)) / 255.0
    hm = (plt.get_cmap('jet')(cam_r)[:, :, :3] * 255).astype(np.uint8)
    return (0.6 * img + 0.4 * hm).clip(0, 255).astype(np.uint8)


# =============================================================================
#  14. PREPARAR DIRECTORIOS
# =============================================================================
def prepare_directories(prep_train, prep_val, prep_test, classes_used, prep_dir):
    """
    Copia las imágenes desde Drive a la RAM local en una estructura de carpetas
    train/val/test por clase. La copia local elimina la latencia de red por batch
    durante el entrenamiento. El contenido previo se borra para evitar mezclar
    imágenes de sesiones anteriores.
    """
    # Copia las imágenes a RAM local (/content/data/) para acelerar la carga
    # durante el entrenamiento. Leer directamente desde Drive en cada batch
    # introduce latencia de red que alarga el entrenamiento considerablemente.
    # Se borra el contenido previo para no mezclar imágenes de sesiones anteriores.
    for split_name, split_data in [('train', prep_train), ('val', prep_val),
                                   ('test',  prep_test)]:
        for cls in classes_used:
            target = Path(prep_dir) / split_name / cls
            target.mkdir(parents=True, exist_ok=True)
            for f in target.glob("*"): f.unlink()
            for src in split_data.get(cls, []): shutil.copy(src, target)
    log("✓ Imágenes copiadas a estructura de carpetas")


# =============================================================================
#  CALIBRACIÓN DE UMBRALES CLÍNICOS
# =============================================================================
def calibrate_threshold(val_probs, val_labels, cls_idx, cls_name,
                        sensitivity_target, specificity_floor,
                        use_fbeta=False, beta=2.0):
    """
    Busca el mejor theta en [0.01, 0.99] con spec >= specificity_floor.

    argmax-theta (use_fbeta=False): selecciona el theta más alto que cumple
    tanto sens >= sensitivity_target como spec >= specificity_floor.
    Elegir el theta más alto (más conservador) minimiza los falsos positivos
    adicionales introducidos por el umbral.

    Fallback en dos niveles para mayor robustez:
      1. Si no hay candidatos con spec >= specificity_floor, relaja el floor
         a 0.50 y busca de nuevo. Avisa en el log. Esto evita devolver 0.5
         ciegamente en sesiones con modelos ligeramente más débiles. Con
         buenos modelos este fallback nunca se activa.
      2. Si tampoco hay candidatos, devuelve 0.5 (argmax estándar).
    """
    def _search(floor):
        cands = []
        for theta in np.linspace(0.01, 0.99, 199):
            preds = (val_probs[:, cls_idx] >= theta).astype(int)
            lbin  = (val_labels == cls_idx).astype(int)
            tp = ((preds == 1) & (lbin == 1)).sum()
            fp = ((preds == 1) & (lbin == 0)).sum()
            fn = ((preds == 0) & (lbin == 1)).sum()
            tn = ((preds == 0) & (lbin == 0)).sum()
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            if spec < floor:
                continue
            if use_fbeta:
                score = (1 + beta**2) * sens * spec / (beta**2 * spec + sens + 1e-9)
            else:
                if sens >= sensitivity_target:
                    score = theta
                else:
                    continue
            cands.append((score, theta, sens, spec))
        return cands

    metodo = f"F-beta(beta={beta})" if use_fbeta else "argmax-theta"

    candidates = _search(specificity_floor)
    if candidates:
        _, best_theta, best_sens, best_spec = max(candidates, key=lambda x: x[0])
        log(f"  {cls_name}: theta={best_theta:.3f} | sens={best_sens:.4f} | "
            f"spec={best_spec:.4f} | metodo={metodo} | floor={specificity_floor:.2f}")
        return best_theta

    fallback_floor = 0.50
    log(f"  aviso {cls_name}: sin candidatos con spec>={specificity_floor:.2f} "
        f"-- fallback con floor={fallback_floor:.2f}")
    candidates_r = _search(fallback_floor)
    if candidates_r:
        _, best_theta, best_sens, best_spec = max(candidates_r, key=lambda x: x[0])
        log(f"  {cls_name}: theta={best_theta:.3f} | sens={best_sens:.4f} | "
            f"spec={best_spec:.4f} | metodo={metodo} | floor=RELAJADO")
        return best_theta

    log(f"  aviso {cls_name}: sens>={sensitivity_target:.2f} inalcanzable "
        f"-- usando 0.5 (argmax estandar)")
    return 0.5


def apply_clinical_thresholds(probs, labels, thresholds, classes_used, priority_order):
    """
    Aplica umbrales por clase maligna en orden inverso de prioridad.
    Si una imagen supera el umbral de MEL y AKIEC a la vez, prevalece MEL
    porque tiene mayor gravedad clínica potencial.
    """
    preds = probs.argmax(axis=1).copy()
    for cls_name in reversed(priority_order):
        if cls_name not in thresholds or cls_name not in classes_used:
            continue
        idx = classes_used.index(cls_name)
        preds[probs[:, idx] > thresholds[cls_name]] = idx
    return preds, balanced_accuracy_score(labels, preds)


def compute_per_class_metrics(preds, labels, classes_used):
    """
    Calcula TP, FP, FN, TN, sensibilidad y especificidad para cada clase.

    Args:
        preds:        array (N,) de predicciones
        labels:       array (N,) de etiquetas verdaderas
        classes_used: lista ordenada de nombres de clase

    Returns:
        DataFrame con una fila por clase y columnas clínicas estándar.
    """
    cm = confusion_matrix(labels, preds)
    TP = np.diag(cm); FP = cm.sum(0) - TP; FN = cm.sum(1) - TP
    TN = cm.sum() - (TP + FP + FN)
    return pd.DataFrame({
        'Clase': classes_used, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'Sensibilidad':  np.round(TP / np.maximum(TP + FN, 1), 4),
        'Especificidad': np.round(TN / np.maximum(TN + FP, 1), 4),
    })


# =============================================================================
#  15. PIPELINE PRINCIPAL
# =============================================================================
def main():
    """
    Pipeline principal de entrenamiento, inferencia y evaluación.

    Gestiona los tres modos de ejecución (entrenamiento completo,
    reanudación con checkpoints, y recalibración de umbrales solo),
    y genera el PDF de resultados y el archivo ZIP de salida.
    El modo activo se determina por las variables globales
    LOAD_TTA_FROM_DIR y RESUME_FROM_CHECKPOINTS.
    """
    global _current_out_dir

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if LOAD_TTA_FROM_DIR:
        OUT_DIR  = os.path.join(OUTPUTS_ROOT, ts + "_thresh")
        mode_str = f"SOLO UMBRALES — arrays de: {LOAD_TTA_FROM_DIR}"
    elif RESUME_FROM_CHECKPOINTS and RESUME_DIR:
        OUT_DIR  = os.path.join(OUTPUTS_ROOT, ts + "_r")
        mode_str = f"REANUDAR (nueva carpeta _r) — checkpoints de: {RESUME_DIR}"
    else:
        OUT_DIR  = os.path.join(OUTPUTS_ROOT, ts)
        mode_str = "Entrenamiento completo desde cero"

    _current_out_dir = OUT_DIR
    os.makedirs(OUTPUTS_ROOT, exist_ok=True)
    os.makedirs(OUT_DIR,      exist_ok=True)

    # En reanudación, copiar los checkpoints y el split al nuevo directorio _r.
    # El split_assignment.json es crítico: garantiza que FORCE_RETRAIN usa
    # exactamente las mismas imágenes que los modelos conservados.
    # Sin esto, un orden de glob distinto en la nueva sesión produce un split
    # diferente, lo que inflaría artificialmente el Test BACC (data leakage).
    if RESUME_FROM_CHECKPOINTS and RESUME_DIR and os.path.isdir(RESUME_DIR):
        copy_exts = ('.pth', '_meta.json', '_history.json')
        copy_names = ('split_assignment.json',)
        for fname in os.listdir(RESUME_DIR):
            if fname.endswith(copy_exts) or fname in copy_names:
                s_ = os.path.join(RESUME_DIR, fname)
                d_ = os.path.join(OUT_DIR, fname)
                if not os.path.exists(d_):
                    shutil.copy2(s_, d_)
        if not os.path.exists(os.path.join(OUT_DIR, 'split_assignment.json')):
            print(f"⚠ split_assignment.json NO encontrado en {RESUME_DIR}\n"
                  f"  El split se recalculará con la semilla fija. Si el código\n"
                  f"  usa ordenación determinista, el nuevo split será idéntico\n"
                  f"  al original. De lo contrario hay riesgo de data leakage.",
                  flush=True)

    # Guardar el script ejecutado en la carpeta de salida para tener
    # trazabilidad exacta de qué código produjo cada resultado.
    try:
        script_dst = os.path.join(OUT_DIR, "CodigoEnPruebaDef_ejecutado.py")
        try:
            shutil.copy2(__file__, script_dst)
        except NameError:
            _tmp_scripts = sorted(
                glob.glob("/tmp/ipykernel_*/*.py"),
                key=os.path.getmtime, reverse=True
            )
            if _tmp_scripts:
                shutil.copy2(_tmp_scripts[0], script_dst)
            else:
                script_dst = None
        if script_dst and os.path.exists(script_dst):
            print(f"Script guardado: {os.path.basename(script_dst)}", flush=True)
    except Exception as _se:
        print(f"No se pudo guardar el script: {_se}", flush=True)

    log_path = os.path.join(OUT_DIR, "ejecucion_log.txt")
    tee = Tee(log_path)
    sys.stdout = tee

    try:
        log("\n" + "="*60)
        log("  CONFIGURACIÓN DE ESTA EJECUCIÓN")
        log("="*60)
        log(f"  Dataset:       {USE_PERCENT*100:.0f}% ({USE_PERCENT})")
        log(f"  Split:         Train {TRAIN_RATIO*100:.0f}% / "
            f"Val {VAL_RATIO*100:.0f}% / Test {TEST_RATIO*100:.0f}%")
        log(f"  Max epochs:    {MAX_EPOCHS}  |  ES patience: "
            f"resnet50={ES_PATIENCE['resnet50']} "
            f"densenet121={ES_PATIENCE['densenet121']} "
            f"efficientnet_b3={ES_PATIENCE['efficientnet_b3']}")
        log(f"  ES min delta:  resnet50={ES_MIN_DELTA['resnet50']} | "
            f"densenet121={ES_MIN_DELTA['densenet121']} | "
            f"efficientnet_b3={ES_MIN_DELTA['efficientnet_b3']}")
        log(f"  TTA rondas:    {TTA_ROUNDS}")
        log(f"  Semilla:       RNG_SEED = {RNG_SEED}")
        log(f"  Modelos:")
        for n in ['resnet50', 'densenet121', 'efficientnet_b3']:
            log(f"    {n}: lr={LR[n]} wd={WEIGHT_DECAY[n]} "
                f"w={ES_BACC_WEIGHTS[n]} p={ES_PATIENCE[n]} "
                f"δ={ES_MIN_DELTA[n]} sched={SCHED_PATIENCE[n]} "
                f"{IMG_SIZE[n]}px bs={BATCH_SIZE[n]}")
        log(f"  Loss: FocalLoss per-model "
            f"(resnet50 γ={FOCAL_GAMMA['resnet50']}, "
            f"densenet121 γ={FOCAL_GAMMA['densenet121']}, "
            f"efficientnet_b3 γ={FOCAL_GAMMA['efficientnet_b3']} [=CE])")
        log(f"  Aug boost clínico: "
            f"{ {k: f'min nivel {v}' for k, v in CLINICAL_AUG_BOOST.items()} }")
        log(f"  Umbrales clínicos:")
        for cls, v in CLINICAL_THRESHOLDS.items():
            metodo = f"F-beta β={v['beta']}" if v['use_fbeta'] else "argmax-θ"
            log(f"    {cls}: sens>={v['sensitivity_target']} "
                f"spec>={v['specificity_floor']} [{metodo}]")
        log(f"  Modo:          {mode_str}")
        log(f"  Carpeta:       {OUT_DIR}")
        log("="*60 + "\n")

        # ==================================================================
        #  RAMA A: SOLO UMBRALES — cargar arrays, saltar entrenamiento
        # ==================================================================
        if LOAD_TTA_FROM_DIR:
            log(f"Cargando arrays de: {LOAD_TTA_FROM_DIR}")
            required = ['tta_sum_probs.npy', 'tta_labels.npy',
                        'tta_val_sum.npy',   'tta_val_labels.npy',
                        'classes_used.json']
            for f in required:
                p = os.path.join(LOAD_TTA_FROM_DIR, f)
                assert os.path.exists(p), f"Archivo no encontrado: {p}"

            tta_sum_probs  = np.load(os.path.join(LOAD_TTA_FROM_DIR, 'tta_sum_probs.npy'))
            tta_labels     = np.load(os.path.join(LOAD_TTA_FROM_DIR, 'tta_labels.npy'))
            tta_val_sum    = np.load(os.path.join(LOAD_TTA_FROM_DIR, 'tta_val_sum.npy'))
            tta_val_labels = np.load(os.path.join(LOAD_TTA_FROM_DIR, 'tta_val_labels.npy'))

            with open(os.path.join(LOAD_TTA_FROM_DIR, 'classes_used.json')) as f:
                classes_used = json.load(f)

            orig_thresh_path = os.path.join(LOAD_TTA_FROM_DIR, 'calibrated_thresholds.json')
            original_thresholds = {}
            if os.path.exists(orig_thresh_path):
                with open(orig_thresh_path) as f:
                    original_thresholds = json.load(f)

            log(f"✓ Arrays cargados — Test: {len(tta_labels)} muestras | "
                f"Val: {len(tta_val_labels)} muestras")
            log(f"  Clases: {classes_used}")
            if original_thresholds:
                log(f"  Umbrales originales (referencia): {original_thresholds}")

            _mn_all = ['resnet50', 'densenet121', 'efficientnet_b3']
            _all_hist, _m_times, _m_vbacc, _m_tbacc, _m_epoch, _m_epot = {}, {}, {}, {}, {}, {}
            for _nm in _mn_all:
                _hp = os.path.join(LOAD_TTA_FROM_DIR, f'{_nm}_history.json')
                _mp = os.path.join(LOAD_TTA_FROM_DIR, f'{_nm}_meta.json')
                if os.path.exists(_hp):
                    with open(_hp) as _f: _all_hist[_nm] = json.load(_f)
                if os.path.exists(_mp):
                    with open(_mp) as _f: _meta_d = json.load(_f)
                    _m_vbacc[_nm]  = _meta_d['val_bacc']
                    _m_tbacc[_nm]  = _meta_d.get('test_bacc', 0.0)
                    _m_times[_nm]  = _meta_d['tiempo_segundos']
                    _m_epoch[_nm]  = _meta_d['best_epoch']
                    _m_epot[_nm]   = (len(_all_hist[_nm]['val_bacc'])
                                      if _nm in _all_hist else _meta_d['best_epoch'])

            _cc_path = os.path.join(LOAD_TTA_FROM_DIR, 'class_counts.json')
            _class_counts_l = _total_tr = _total_va = _total_te = _total_im = None
            if os.path.exists(_cc_path):
                with open(_cc_path) as _f: _cc_raw = json.load(_f)
                _class_counts_l = {c: tuple(v) for c, v in _cc_raw.items()}
            else:
                try:
                    _lm_cc = merge_groundtruth_csvs(CSV_DIR)
                    _sp_cc = json.load(
                        open(os.path.join(LOAD_TTA_FROM_DIR, 'split_assignment.json')))
                    _cc_tmp = {c: [0, 0, 0] for c in classes_used}
                    for _iid, _spl in _sp_cc.items():
                        if _iid in _lm_cc and _lm_cc[_iid] in _cc_tmp:
                            _si = {'train': 0, 'val': 1, 'test': 2}.get(_spl, -1)
                            if _si >= 0:
                                _cc_tmp[_lm_cc[_iid]][_si] += 1
                    _class_counts_l = {c: tuple(v) for c, v in _cc_tmp.items()}
                    log("✓ Distribución de clases reconstruida desde split_assignment.json")
                except Exception as _ecc:
                    log(f"  ⚠ No se pudo reconstruir distribución: {_ecc}")
            if _class_counts_l:
                _total_tr = sum(v[0] for v in _class_counts_l.values())
                _total_va = sum(v[1] for v in _class_counts_l.values())
                _total_te = sum(v[2] for v in _class_counts_l.values())
                _total_im = _total_tr + _total_va + _total_te

            _summary_df_l = None
            _best_model_l = None
            _has_full     = bool(_m_vbacc)
            if _has_full:
                _best_model_l = max(_m_vbacc, key=_m_vbacc.get)
                _rows_l = []
                for _nm in _mn_all:
                    if _nm not in _m_vbacc: continue
                    _mi, _si = divmod(int(_m_times[_nm]), 60)
                    _rows_l.append({
                        'Modelo': _nm, 'Val BACC': round(_m_vbacc[_nm], 4),
                        'Test BACC': round(_m_tbacc[_nm], 4),
                        'Mejor epoch': _m_epoch[_nm],
                        'Epochs totales': _m_epot[_nm],
                        'Tiempo': f'{_mi}m {_si}s',
                    })
                _summary_df_l = pd.DataFrame(_rows_l) if _rows_l else None
                log(f"✓ Datos de entrenamiento cargados — PDF con curvas y tabla")

            _trained_models_l = {}
            _test_dataset_l   = None
            if _has_full:
                _pth_ok = all(
                    os.path.exists(os.path.join(LOAD_TTA_FROM_DIR, f'{_nm}_best.pth'))
                    for _nm in _mn_all if _nm in _m_vbacc
                )
                if _pth_ok:
                    try:
                        _label_map_l = merge_groundtruth_csvs(CSV_DIR)
                        _split_l = json.load(
                            open(os.path.join(LOAD_TTA_FROM_DIR, 'split_assignment.json'))
                        )
                        _prep_test_l = {c: [] for c in classes_used}
                        for _iid, _sp in _split_l.items():
                            if _sp == 'test' and _iid in _label_map_l:
                                _cls_l = _label_map_l[_iid]
                                if _cls_l in _prep_test_l:
                                    _fpath = os.path.join(SOURCE_DIR, _iid + '.jpg')
                                    if os.path.exists(_fpath):
                                        _prep_test_l[_cls_l].append(_fpath)
                        for _cls_l in classes_used:
                            _tgt_l = Path(PREP_DIR) / 'test' / _cls_l
                            _tgt_l.mkdir(parents=True, exist_ok=True)
                            for _f in _tgt_l.glob("*"): _f.unlink()
                            for _src_f in _prep_test_l.get(_cls_l, []):
                                shutil.copy(_src_f, _tgt_l)
                        _ts_l = build_transforms(IMG_SIZE[_best_model_l])
                        _test_dataset_l = SkinDataset(
                            str(Path(PREP_DIR) / 'test'), classes_used,
                            is_train=False, transforms=_ts_l
                        )
                        for _nm in _mn_all:
                            if _nm not in _m_vbacc: continue
                            _pth = os.path.join(LOAD_TTA_FROM_DIR, f'{_nm}_best.pth')
                            _m = build_model(_nm, len(classes_used))
                            _m.load_state_dict(
                                torch.load(_pth, weights_only=False, map_location=device))
                            _m.eval()
                            _trained_models_l[_nm] = _m
                        log(f"✓ Modelos cargados ({len(_trained_models_l)}) — "
                            f"Grad-CAM e imágenes representativas disponibles")
                        try:
                            _bm_eval = _trained_models_l[_best_model_l]
                            _eval_loader_l = DataLoader(
                                _test_dataset_l,
                                batch_size=BATCH_SIZE[_best_model_l],
                                shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=False)
                            _, _, _te_preds_l, _te_labels_l = evaluate(
                                _bm_eval, _eval_loader_l, nn.CrossEntropyLoss(),
                                classes_used)
                            _final_cm_l   = confusion_matrix(_te_labels_l, _te_preds_l)
                            _metrics_df_l = compute_per_class_metrics(
                                _te_preds_l, _te_labels_l, classes_used)
                            log(f"✓ CM y métricas de {_best_model_l} disponibles")
                        except Exception as _eev:
                            log(f"  ⚠ Evaluación individual omitida: {_eev}")
                    except Exception as _elo:
                        log(f"  ⚠ No se pudieron cargar modelos para Grad-CAM: {_elo}")
                else:
                    log("  Sin .pth en carpeta fuente — Grad-CAM no disponible")
            else:
                log("  Sin meta.json — PDF reducido (sin curvas de entrenamiento)")

            tta_preds = tta_sum_probs.argmax(axis=1)
            tta_bacc  = balanced_accuracy_score(tta_labels, tta_preds)
            mel_idx   = classes_used.index('MEL') if 'MEL' in classes_used else -1
            if mel_idx >= 0:
                imt = (tta_labels == mel_idx).astype(int)
                imp = (tta_preds  == mel_idx).astype(int)
                tp = ((imp == 1) & (imt == 1)).sum()
                fn = ((imp == 0) & (imt == 1)).sum()
                tn = ((imp == 0) & (imt == 0)).sum()
                fp = ((imp == 1) & (imt == 0)).sum()
                mel_sens_baseline = tp / max(tp + fn, 1)
                mel_spec_baseline = tn / max(tn + fp, 1)
            else:
                mel_sens_baseline = mel_spec_baseline = 0.0

            metrics_argmax_df = compute_per_class_metrics(tta_preds, tta_labels, classes_used)
            malignant_classes = [c for c in ['MEL', 'BCC', 'AKIEC'] if c in classes_used]
            bacc_malignas_argmax = float(
                metrics_argmax_df[metrics_argmax_df['Clase'].isin(malignant_classes)]['Sensibilidad'].mean()
            )

            log("\n===== CALIBRACIÓN DE UMBRALES (sobre val cargado) =====")
            calibrated_thresholds = {}
            for cls_name, targets in CLINICAL_THRESHOLDS.items():
                if cls_name not in classes_used:
                    log(f"  {cls_name}: no en classes_used — omitida"); continue
                theta = calibrate_threshold(
                    val_probs=tta_val_sum, val_labels=tta_val_labels,
                    cls_idx=classes_used.index(cls_name), cls_name=cls_name,
                    sensitivity_target=targets['sensitivity_target'],
                    specificity_floor=targets['specificity_floor'],
                    use_fbeta=targets['use_fbeta'], beta=targets['beta'],
                )
                calibrated_thresholds[cls_name] = theta

            tta_thresh_preds, tta_thresh_bacc = apply_clinical_thresholds(
                probs=tta_sum_probs, labels=tta_labels,
                thresholds=calibrated_thresholds, classes_used=classes_used,
                priority_order=THRESHOLD_PRIORITY,
            )
            metrics_thresh_df = compute_per_class_metrics(
                tta_thresh_preds, tta_labels, classes_used
            )
            bacc_malignas_thresh = float(
                metrics_thresh_df[metrics_thresh_df['Clase'].isin(malignant_classes)]['Sensibilidad'].mean()
            )

            theta_label = "  ".join(
                f"theta_{c}={calibrated_thresholds[c]:.3f}"
                for c in THRESHOLD_PRIORITY if c in calibrated_thresholds
            )
            if _summary_df_l is not None:
                _theta_short = "  ".join(
                    f"{c}={calibrated_thresholds[c]:.3f}"
                    for c in THRESHOLD_PRIORITY if c in calibrated_thresholds
                )
                _tta_rows = pd.DataFrame([
                    {'Modelo': 'TTA Ensemble', 'Val BACC': '-',
                     'Test BACC': round(tta_bacc, 4), 'Mejor epoch': '-',
                     'Epochs totales': '-', 'Tiempo': '-'},
                    {'Modelo': 'TTA + umbrales clínicos', 'Val BACC': '-',
                     'Test BACC': round(tta_thresh_bacc, 4), 'Mejor epoch': '-',
                     'Epochs totales': _theta_short, 'Tiempo': '-'},
                ])
                _summary_df_l = pd.concat([_summary_df_l, _tta_rows], ignore_index=True)

            log(f"\n===== COMPARATIVA (test) =====")
            log(f"  TTA Ensemble (argmax):            BACC = {tta_bacc:.4f}  "
                f"MEL sens = {mel_sens_baseline:.3f}  spec = {mel_spec_baseline:.3f}")
            log(f"  TTA Ensemble (umbrales nuevos):   BACC = {tta_thresh_bacc:.4f}  "
                f"[{theta_label}]")
            log(f"  Delta BACC global: {tta_thresh_bacc - tta_bacc:+.4f}")
            log(f"  BACC malignas: Argmax={bacc_malignas_argmax:.4f} → "
                f"Umbrales={bacc_malignas_thresh:.4f} "
                f"({bacc_malignas_thresh - bacc_malignas_argmax:+.4f})")
            log("  Métricas por clase con umbrales:")
            for _, row in metrics_thresh_df.iterrows():
                marker = " <-- crítico" if row['Clase'] in CLINICAL_THRESHOLDS else ""
                log(f"    {row['Clase']:6s}  sens={row['Sensibilidad']:.4f}  "
                    f"spec={row['Especificidad']:.4f}{marker}")

            with open(os.path.join(OUT_DIR, 'calibrated_thresholds.json'), 'w') as f:
                json.dump(calibrated_thresholds, f, indent=2)
            with open(os.path.join(OUT_DIR, 'classes_used.json'), 'w') as f:
                json.dump(classes_used, f)
            metrics_thresh_df.to_csv(
                os.path.join(OUT_DIR, 'metricas_clinicas_umbrales.csv'), index=False
            )
            metrics_argmax_df.to_csv(
                os.path.join(OUT_DIR, 'metricas_tta_argmax.csv'), index=False
            )

            _results_thresh_json = {
                "run": {
                    "seed":        RNG_SEED,
                    "timestamp":   ts,
                    "environment": _ENV,
                    "mode":        "threshold_only",
                    "source_dir":  str(LOAD_TTA_FROM_DIR),
                    "classes":     classes_used,
                },
                "config": {"tta_rounds": TTA_ROUNDS},
                "tta_ensemble": {"rounds": TTA_ROUNDS, "test_bacc": round(tta_bacc, 4)},
                "clinical": {
                    "thresholds":     {k: round(v, 4) for k, v in calibrated_thresholds.items()},
                    "test_bacc":      round(tta_thresh_bacc, 4),
                    "bacc_malignant": round(bacc_malignas_thresh, 4),
                    "per_class": {
                        row["Clase"]: {
                            "sensitivity": round(float(row["Sensibilidad"]), 4),
                            "specificity": round(float(row["Especificidad"]), 4),
                        }
                        for _, row in metrics_thresh_df.iterrows()
                    },
                },
            }
            with open(os.path.join(OUT_DIR, "results.json"), "w") as _rjf:
                json.dump(_results_thresh_json, _rjf, indent=2)
            log("\u2713 results.json guardado")

            generate_results_pdf(
                out_dir=OUT_DIR, classes_used=classes_used,
                class_counts=_class_counts_l,
                total_train=_total_tr, total_val=_total_va,
                total_test=_total_te, total_imgs=_total_im,
                all_histories=_all_hist if _has_full else None,
                model_names=list(_m_vbacc.keys()) if _has_full else None,
                best_model_name=_best_model_l,
                final_cm=_final_cm_l   if '_final_cm_l'   in dir() else None,
                metrics_df=_metrics_df_l if '_metrics_df_l' in dir() else None,
                metrics_argmax_df=metrics_argmax_df,
                final_probs=None, final_labels=None, ens_bacc=None,
                summary_df=_summary_df_l,
                trained_models=_trained_models_l if _trained_models_l else None,
                test_dataset=_test_dataset_l   if '_test_dataset_l' in dir() else None,
                model_times=_m_times if _has_full else None,
                calibrated_thresholds=calibrated_thresholds,
                tta_bacc=tta_bacc, tta_thresh_bacc=tta_thresh_bacc,
                tta_sum_probs=tta_sum_probs, tta_labels=tta_labels,
                tta_thresh_preds=tta_thresh_preds,
                metrics_thresh_df=metrics_thresh_df,
                mel_sens_baseline=mel_sens_baseline,
                mel_spec_baseline=mel_spec_baseline,
                bacc_malignas_argmax=bacc_malignas_argmax,
                bacc_malignas_thresh=bacc_malignas_thresh,
                malignant_classes=malignant_classes,
                tta_val_labels=tta_val_labels,
                threshold_only=False,
                original_thresholds=original_thresholds,
                load_tta_source=LOAD_TTA_FROM_DIR,
            )
            log(f"\n✓ PDF generado: {os.path.join(OUT_DIR, 'resultados_tfg.pdf')}")
            log(f"✓ Outputs en: {OUT_DIR}")
            try:
                _zip_name = os.path.basename(OUT_DIR)
                _zip_base = os.path.join(OUTPUTS_ROOT, _zip_name)
                _zip_full = shutil.make_archive(
                    base_name=_zip_base, format='zip',
                    root_dir=OUTPUTS_ROOT, base_dir=_zip_name,
                )
                log(f"✓ ZIP creado: {_zip_full}  ({os.path.getsize(_zip_full)/1e6:.1f} MB)")
            except Exception as _ze:
                log(f"⚠ No se pudo crear el ZIP: {_ze}")
            return

        # ==================================================================
        #  RAMA B: ENTRENAMIENTO (completo o reanudación en carpeta _r)
        # ==================================================================
        label_map = merge_groundtruth_csvs(CSV_DIR)
        id2path   = index_images(SOURCE_DIR)
        id2path   = {k: v for k, v in id2path.items() if k in label_map}

        # Si existe split guardado de una ejecución anterior, lo cargo
        # para garantizar exactamente el mismo train/val/test. En modo
        # RESUME fue copiado desde RESUME_DIR al inicio de main().
        split_file = os.path.join(OUT_DIR, 'split_assignment.json')
        if os.path.exists(split_file):
            log("✓ Cargando split desde split_assignment.json (mismos sets que run original)")
            prep_train, prep_val, prep_test = load_split_from_file(
                split_file, id2path, label_map
            )
        else:
            prep_train, prep_val, prep_test = split_dataset(
                label_map, id2path,
                train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, use_percent=USE_PERCENT,
                save_path=split_file,
            )

        classes_used = [
            c for c in CLASSES
            if len(prep_train.get(c, [])) >= MIN_SAMPLES
            and len(prep_val.get(c,   [])) >= MIN_SAMPLES
            and len(prep_test.get(c,  [])) >= MIN_SAMPLES
        ]
        excluded = [c for c in CLASSES if c not in classes_used]
        if excluded:
            log(f"⚠ Clases excluidas por pocas muestras: {excluded}")
        log(f"✓ Clases usadas: {classes_used}")

        prepare_directories(prep_train, prep_val, prep_test, classes_used, PREP_DIR)

        class_counts = {
            c: (len(prep_train.get(c, [])), len(prep_val.get(c, [])),
                len(prep_test.get(c, [])))
            for c in classes_used
        }
        total_train = sum(v[0] for v in class_counts.values())
        total_val   = sum(v[1] for v in class_counts.values())
        total_test  = sum(v[2] for v in class_counts.values())
        total_imgs  = total_train + total_val + total_test
        with open(os.path.join(OUT_DIR, 'class_counts.json'), 'w') as _ccf:
            json.dump({c: list(v) for c, v in class_counts.items()}, _ccf)

        log("\n===== DISTRIBUCIÓN DEL DATASET =====")
        for c in classes_used:
            tr, va, te = class_counts[c]
            log(f"  {c:8s} {CLASS_NAMES_FULL[c]:50s} "
                f"Train: {tr:5d} | Val: {va:4d} | Test: {te:5d}")

        model_names           = ['resnet50', 'densenet121', 'efficientnet_b3']
        trained_models        = {}
        model_best_baccs      = {}
        all_histories         = {}
        model_times           = {}
        es_trackers           = {}
        trained_test_loaders  = {}
        trained_test_datasets = {}
        trained_val_datasets  = {}
        model_test_baccs      = {}
        # CrossEntropyLoss sin smoothing para val/test: con smoothing la
        # pérdida es artificialmente alta y el scheduler bajaba el LR antes
        # de tiempo. Sin smoothing la pérdida es directamente interpretable.
        eval_criterion = nn.CrossEntropyLoss()

        for name in model_names:
            best_ckpt = os.path.join(OUT_DIR, f"{name}_best.pth")
            meta_path = os.path.join(OUT_DIR, f"{name}_meta.json")
            hist_path = os.path.join(OUT_DIR, f"{name}_history.json")

            mt       = build_transforms(IMG_SIZE[name])
            mb       = BATCH_SIZE[name]
            val_ds   = SkinDataset(Path(PREP_DIR)/'val',  classes_used,
                                   is_train=False, transforms=mt)
            test_ds  = SkinDataset(Path(PREP_DIR)/'test', classes_used,
                                   is_train=False, transforms=mt)
            val_loader  = DataLoader(val_ds,  batch_size=mb, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=False)
            test_loader = DataLoader(test_ds, batch_size=mb, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=False)
            trained_test_loaders[name]  = test_loader
            trained_test_datasets[name] = test_ds
            trained_val_datasets[name]  = val_ds

            # Si existe checkpoint con su meta.json, se carga directamente.
            # FORCE_RETRAIN permite reentrenar modelos concretos aunque
            # tengan checkpoint, útil cuando una sesión anterior produjo
            # una convergencia anómala por variabilidad del entorno GPU.
            if (os.path.exists(best_ckpt) and os.path.exists(meta_path)
                    and name not in FORCE_RETRAIN):
                log(f"\n{'='*60}")
                log(f"  Cargando checkpoint: {name}")
                log(f"{'='*60}")
                model = build_model(name, len(classes_used))
                model.load_state_dict(
                    torch.load(best_ckpt, weights_only=False, map_location=device)
                )
                trained_models[name] = model
                with open(meta_path) as f:
                    meta = json.load(f)
                model_best_baccs[name] = meta['val_bacc']
                model_times[name]      = meta['tiempo_segundos']
                es_trackers[name]      = _CheckpointMeta(meta, ES_BACC_WEIGHTS[name])
                all_histories[name]    = (json.load(open(hist_path))
                                          if os.path.exists(hist_path) else None)
                mins, secs = divmod(int(meta['tiempo_segundos']), 60)
                log(f"  ✓ {name} — Val BACC: {meta['val_bacc']:.4f} | "
                    f"Epoch: {meta['best_epoch']} | "
                    f"Tiempo original: {mins}m {secs}s")
                # El test_bacc guardado en meta.json es el valor definitivo.
                # Se usa el valor guardado para no re-evaluar sobre el test set,
                # que debe permanecer intocable hasta la evaluación final.
                if 'test_bacc' in meta:
                    model_test_baccs[name] = meta['test_bacc']
                    log(f"  — Test BACC (meta.json original): {meta['test_bacc']:.4f}")
                else:
                    # Fallback para checkpoints guardados sin campo test_bacc
                    _, _, test_bacc_chk, _, _, _ = evaluate(
                        model, test_loader, eval_criterion, classes_used
                    )
                    model_test_baccs[name] = test_bacc_chk
                    log(f"  — Test BACC (recalculado): {test_bacc_chk:.4f}")
                continue

            log(f"\n{'='*60}")
            log(f"  Entrenando: {name}")
            log(f"{'='*60}")
            model   = build_model(name, len(classes_used))
            t_start = time.time()

            train_ds = SkinDataset(Path(PREP_DIR)/'train', classes_used,
                                   is_train=True, transforms=mt)
            sampler      = build_weighted_sampler(train_ds)
            train_loader = DataLoader(train_ds, batch_size=mb, sampler=sampler,
                                      num_workers=NUM_WORKERS, pin_memory=False)
            log(f"  Resolución: {IMG_SIZE[name]}px | Batch: {mb}")

            # FocalLoss con smoothing solo en entrenamiento — ver docstring.
            train_crit = FocalLoss(gamma=FOCAL_GAMMA[name], label_smoothing=0.1)
            es = EarlyStopping(patience=ES_PATIENCE[name],
                               min_delta=ES_MIN_DELTA[name],
                               checkpoint_path=best_ckpt,
                               bacc_weight=ES_BACC_WEIGHTS[name])
            history   = {k: [] for k in ('train_loss', 'train_acc', 'train_bacc',
                                          'val_loss',   'val_acc',   'val_bacc')}
            optimizer = optim.Adam(model.parameters(),
                                   lr=LR[name], weight_decay=WEIGHT_DECAY[name])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=SCHED_PATIENCE[name]
            )

            for epoch in range(1, MAX_EPOCHS + 1):
                tr_loss, tr_acc, tr_bacc = train_one_epoch(
                    model, train_loader, train_crit, optimizer
                )
                va_loss, va_acc, va_bacc, _, _, _ = evaluate(
                    model, val_loader, eval_criterion, classes_used
                )
                # El scheduler recibe la puntuación negada porque ReduceLROnPlateau
                # busca un mínimo, pero queremos maximizar el score del ES.
                scheduler.step(-es._score(va_bacc, va_loss))
                for k, v in zip(
                    ['train_loss','train_acc','train_bacc',
                     'val_loss','val_acc','val_bacc'],
                    [tr_loss, tr_acc, tr_bacc, va_loss, va_acc, va_bacc]
                ):
                    history[k].append(v)
                log(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
                    f"Train Loss: {tr_loss:.4f}  BACC: {tr_bacc:.4f} | "
                    f"Val Loss: {va_loss:.4f}  BACC: {va_bacc:.4f}")
                if es.step(va_bacc, va_loss, model, epoch):
                    log(f"  ⏹ Early stopping — mejor epoch: {es.best_epoch}")
                    break

            # Cargar el mejor checkpoint guardado por el early stopping,
            # no el estado del último epoch que puede haber sobreajustado.
            model.load_state_dict(
                torch.load(best_ckpt, weights_only=False, map_location=device)
            )
            trained_models[name]   = model
            model_best_baccs[name] = es.best_bacc
            all_histories[name]    = history
            es_trackers[name]      = es

            elapsed = time.time() - t_start
            model_times[name] = elapsed
            mins, secs = divmod(int(elapsed), 60)
            log(f"  ✓ Mejor checkpoint — epoch {es.best_epoch} | "
                f"Val BACC: {es.best_bacc:.4f} | Val Loss: {es.best_loss:.4f} | "
                f"Score={es._score(es.best_bacc, es.best_loss):.4f} | "
                f"Tiempo: {mins}m {secs}s")

            _, _, test_bacc_live, _, _, _ = evaluate(
                model, test_loader, eval_criterion, classes_used
            )
            model_test_baccs[name] = test_bacc_live
            log(f"  — Test BACC: {test_bacc_live:.4f}")

            with open(meta_path, 'w') as f:
                json.dump({'val_bacc':         es.best_bacc,
                           'val_loss':         es.best_loss,
                           'test_bacc':        test_bacc_live,
                           'best_epoch':       es.best_epoch,
                           'tiempo_segundos':  elapsed}, f, indent=2)
            with open(hist_path, 'w') as f:
                json.dump(history, f)
            log(f"  ✓ Guardados: {name}_best.pth | _meta.json | _history.json")

        # ── Ensemble ponderado ────────────────────────────────────────────
        # Cada modelo recibe un peso proporcional a su val BACC.
        # Se usa val BACC y no test BACC porque el test set tiene que
        # permanecer completamente aislado — ninguna decisión de diseño
        # puede depender de él sin contaminar la evaluación final.
        log("\n===== ENSEMBLE PONDERADO (test) =====")
        total_bacc_sum = sum(model_best_baccs.values())
        log("  Pesos (proporcionales a Val BACC):")
        for n, bacc in model_best_baccs.items():
            peso = bacc / total_bacc_sum
            log(f"    {n:20s}  Val BACC: {bacc:.4f}  — peso: {peso:.4f} ({peso*100:.1f}%)")

        ens_sum_probs = None; ens_labels_all = None
        for n, model in trained_models.items():
            w = model_best_baccs[n] / total_bacc_sum
            model.eval(); bp, bl = [], []
            with torch.no_grad():
                for x, y in trained_test_loaders[n]:
                    bp.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
                    bl.extend(y.numpy())
            mp = np.vstack(bp) * w
            if ens_sum_probs is None: ens_sum_probs = mp; ens_labels_all = bl
            else: ens_sum_probs += mp

        ens_preds = ens_sum_probs.argmax(axis=1)
        ens_bacc  = balanced_accuracy_score(ens_labels_all, ens_preds)
        log(f"✓ Ensemble Test BACC: {ens_bacc:.4f}")

        best_model_name = max(model_best_baccs, key=model_best_baccs.get)
        best_model      = trained_models[best_model_name]
        log(f"\n✓ Mejor modelo individual: {best_model_name} "
            f"(Val BACC={model_best_baccs[best_model_name]:.4f})")

        # ── TTA ───────────────────────────────────────────────────────────
        log("\n===== TEST-TIME AUGMENTATION (test) =====")
        log(f"  TTA individual ({best_model_name}, {TTA_ROUNDS} rondas)...")
        tta_best_probs, tta_best_labels = predict_with_tta(
            best_model, trained_test_datasets[best_model_name],
            TTA_ROUNDS, BATCH_SIZE[best_model_name]
        )
        tta_best_bacc = balanced_accuracy_score(
            tta_best_labels, tta_best_probs.argmax(axis=1)
        )
        log(f"✓ TTA {best_model_name} Test BACC ({TTA_ROUNDS} rondas): {tta_best_bacc:.4f}")

        log(f"\n  TTA ensemble (3 modelos × {TTA_ROUNDS} rondas)...")
        total_bacc_tta = sum(model_best_baccs.values())
        tta_sum_probs  = None; tta_labels = None
        for name, model in trained_models.items():
            w = model_best_baccs[name] / total_bacc_tta
            log(f"    {name} (peso {w*100:.1f}%)...")
            mtp, mtl = predict_with_tta(model, trained_test_datasets[name],
                                        TTA_ROUNDS, BATCH_SIZE[name])
            if tta_sum_probs is None: tta_sum_probs = mtp * w; tta_labels = mtl
            else: tta_sum_probs += mtp * w

        tta_preds = tta_sum_probs.argmax(axis=1)
        tta_bacc  = balanced_accuracy_score(tta_labels, tta_preds)
        log(f"✓ TTA Ensemble Test BACC ({TTA_ROUNDS} rondas × 3 modelos): {tta_bacc:.4f}")

        mel_idx_pre = classes_used.index('MEL')
        imt = (tta_labels == mel_idx_pre).astype(int)
        imp = (tta_preds  == mel_idx_pre).astype(int)
        tp = ((imp==1)&(imt==1)).sum(); fn = ((imp==0)&(imt==1)).sum()
        tn = ((imp==0)&(imt==0)).sum(); fp = ((imp==1)&(imt==0)).sum()
        mel_sens_baseline = tp / max(tp + fn, 1)
        mel_spec_baseline = tn / max(tn + fp, 1)

        log(f"\n  Comparativa TTA:")
        log(f"    Ensemble sin TTA:         {ens_bacc:.4f}")
        log(f"    TTA {best_model_name} solo: {tta_best_bacc:.4f}")
        log(f"    TTA Ensemble:             {tta_bacc:.4f}")
        log(f"    MEL sens (TTA argmax):    {mel_sens_baseline:.4f}")

        # TTA sobre val para calibrar umbrales — los thresholds se ajustan
        # sobre validación y nunca sobre test para evitar sobreajuste.
        log(f"\n  TTA ensemble sobre val ({TTA_ROUNDS} rondas × 3 modelos)...")
        total_bacc_cal = sum(model_best_baccs.values())
        tta_val_sum    = None; tta_val_labels = None
        for name, model in trained_models.items():
            w = model_best_baccs[name] / total_bacc_cal
            vtp, vtl = predict_with_tta(model, trained_val_datasets[name],
                                        TTA_ROUNDS, BATCH_SIZE[name])
            if tta_val_sum is None: tta_val_sum = vtp * w; tta_val_labels = vtl
            else: tta_val_sum += vtp * w

        # ── Calibración de umbrales clínicos ──────────────────────────────
        log("\n===== CALIBRACIÓN DE UMBRALES CLÍNICOS (sobre val) =====")
        calibrated_thresholds = {}
        for cls_name, targets in CLINICAL_THRESHOLDS.items():
            if cls_name not in classes_used:
                log(f"  {cls_name}: no en classes_used — omitida"); continue
            theta = calibrate_threshold(
                val_probs=tta_val_sum, val_labels=tta_val_labels,
                cls_idx=classes_used.index(cls_name), cls_name=cls_name,
                sensitivity_target=targets['sensitivity_target'],
                specificity_floor=targets['specificity_floor'],
                use_fbeta=targets['use_fbeta'], beta=targets['beta'],
            )
            calibrated_thresholds[cls_name] = theta

        tta_thresh_preds, tta_thresh_bacc = apply_clinical_thresholds(
            probs=tta_sum_probs, labels=tta_labels,
            thresholds=calibrated_thresholds, classes_used=classes_used,
            priority_order=THRESHOLD_PRIORITY,
        )
        metrics_thresh_df = compute_per_class_metrics(
            tta_thresh_preds, tta_labels, classes_used
        )
        malignant_classes = [c for c in ['MEL', 'BCC', 'AKIEC'] if c in classes_used]
        preds_argmax      = tta_sum_probs.argmax(axis=1)
        metrics_argmax_df = compute_per_class_metrics(preds_argmax, tta_labels, classes_used)
        bacc_malignas_argmax = float(
            metrics_argmax_df[metrics_argmax_df['Clase'].isin(malignant_classes)]['Sensibilidad'].mean()
        )
        bacc_malignas_thresh = float(
            metrics_thresh_df[metrics_thresh_df['Clase'].isin(malignant_classes)]['Sensibilidad'].mean()
        )
        theta_label = "  ".join(
            f"theta_{c}={calibrated_thresholds[c]:.3f}"
            for c in THRESHOLD_PRIORITY if c in calibrated_thresholds
        )

        log(f"\n===== COMPARATIVA FINAL (test) =====")
        log(f"  TTA Ensemble (argmax):            BACC = {tta_bacc:.4f}  "
            f"MEL sens = {mel_sens_baseline:.3f}  spec = {mel_spec_baseline:.3f}")
        log(f"  TTA Ensemble (umbrales clínicos): BACC = {tta_thresh_bacc:.4f}  "
            f"[{theta_label}]")
        log(f"  Delta BACC global: {tta_thresh_bacc - tta_bacc:+.4f}")
        log(f"  BACC malignas: Argmax={bacc_malignas_argmax:.4f} → "
            f"Umbrales={bacc_malignas_thresh:.4f} "
            f"({bacc_malignas_thresh - bacc_malignas_argmax:+.4f})")
        log("  Métricas por clase con umbrales:")
        for _, row in metrics_thresh_df.iterrows():
            marker = " <-- crítico" if row['Clase'] in CLINICAL_THRESHOLDS else ""
            log(f"    {row['Clase']:6s}  sens={row['Sensibilidad']:.4f}  "
                f"spec={row['Especificidad']:.4f}{marker}")

        _, _, _, final_cm, final_probs, final_labels = evaluate(
            best_model, trained_test_loaders[best_model_name],
            eval_criterion, classes_used
        )
        TP = np.diag(final_cm); FP = final_cm.sum(0) - TP
        FN = final_cm.sum(1) - TP; TN = final_cm.sum() - (TP + FP + FN)
        metrics_df = pd.DataFrame({
            "Clase": classes_used, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "Sensibilidad":  np.round(TP / np.maximum(TP + FN, 1), 4),
            "Especificidad": np.round(TN / np.maximum(TN + FP, 1), 4),
        })
        metrics_df.to_csv(
            os.path.join(OUT_DIR, f"metricas_clinicas_{best_model_name}.csv"), index=False
        )
        metrics_thresh_df.to_csv(
            os.path.join(OUT_DIR, "metricas_clinicas_umbrales_clinicos.csv"), index=False
        )

        summary_rows = []
        for name in model_names:
            mins, secs = divmod(int(model_times[name]), 60)
            summary_rows.append({
                "Modelo": name, "Val BACC": round(model_best_baccs[name], 4),
                "Test BACC": round(model_test_baccs[name], 4),
                "Mejor epoch": es_trackers[name].best_epoch,
                "Epochs totales": (len(all_histories[name]['val_bacc'])
                                   if all_histories[name] else "—"),
                "Tiempo": f"{mins}m {secs}s",
            })
        for label, bacc in [("Ensemble", ens_bacc),
                             (f"TTA ({best_model_name})", tta_best_bacc),
                             ("TTA Ensemble", tta_bacc)]:
            summary_rows.append({
                "Modelo": label, "Val BACC": "-", "Test BACC": round(bacc, 4),
                "Mejor epoch": "-", "Epochs totales": "-", "Tiempo": "-",
            })
        theta_label_short = "  ".join(
            f"θ_{c}={calibrated_thresholds[c]:.3f}"
            for c in THRESHOLD_PRIORITY if c in calibrated_thresholds
        )
        summary_rows.append({
            "Modelo": "TTA + umbrales clinicos", "Val BACC": "-",
            "Test BACC": round(tta_thresh_bacc, 4),
            "Mejor epoch": "-", "Epochs totales": theta_label_short, "Tiempo": "-",
        })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(OUT_DIR, "resumen_modelos.csv"), index=False)

        # results.json: resumen legible por máquina con todos los resultados.
        # Permite comparar ejecuciones sin abrir el PDF.
        _results_json = {
            "run": {
                "seed":         RNG_SEED,
                "timestamp":    ts,
                "environment":  _ENV,
                "dataset":      "HAM10000 / ISIC 2018 Task 3",
                "total_images": total_imgs,
                "split": {"train": total_train, "val": total_val, "test": total_test},
                "classes": classes_used,
            },
            "config": {
                "tta_rounds": TTA_ROUNDS,
                "max_epochs": MAX_EPOCHS,
                "focal_gamma": FOCAL_GAMMA,
                "batch_size":  {k: int(v) for k, v in BATCH_SIZE.items()},
                "lr":          LR,
                "es_patience": ES_PATIENCE,
            },
            "models": {
                name: {
                    "val_bacc":     round(model_best_baccs[name], 4),
                    "test_bacc":    round(model_test_baccs[name], 4),
                    "best_epoch":   es_trackers[name].best_epoch,
                    "total_epochs": (len(all_histories[name]["val_bacc"])
                                     if all_histories[name] else None),
                    "time_min":     round(model_times[name] / 60, 1),
                }
                for name in model_names
            },
            "ensemble":     {"test_bacc": round(ens_bacc, 4)},
            "tta_ensemble": {"rounds": TTA_ROUNDS, "test_bacc": round(tta_bacc, 4)},
            "clinical": {
                "thresholds":     {k: round(v, 4) for k, v in calibrated_thresholds.items()},
                "test_bacc":      round(tta_thresh_bacc, 4),
                "bacc_malignant": round(bacc_malignas_thresh, 4),
                "per_class": {
                    row["Clase"]: {
                        "sensitivity": round(float(row["Sensibilidad"]), 4),
                        "specificity": round(float(row["Especificidad"]), 4),
                    }
                    for _, row in metrics_thresh_df.iterrows()
                },
            },
        }
        with open(os.path.join(OUT_DIR, "results.json"), "w") as _rjf:
            json.dump(_results_json, _rjf, indent=2)
        log("\u2713 results.json guardado")

        cw = {"Modelo": 22, "Val BACC": 10, "Test BACC": 11,
              "Mejor epoch": 13, "Epochs totales": 16, "Tiempo": 9}
        hdr = "".join(c.rjust(w) for c, w in cw.items())
        sep = "-" * len(hdr)
        log("\n" + sep); log(hdr); log(sep)
        for _, row in summary_df.iterrows():
            fmt = {c: (f"{float(v):.4f}" if c in ("Val BACC","Test BACC") and v != "-"
                       else str(v)) for c, v in row.items()}
            log("".join(fmt[c].rjust(w) for c, w in cw.items()))
        log(sep)

        # Guardar arrays de probabilidades TTA para poder recalibrar
        # umbrales en el futuro sin necesidad de reentrenar los modelos.
        np.save(os.path.join(OUT_DIR, 'tta_sum_probs.npy'),    tta_sum_probs)
        np.save(os.path.join(OUT_DIR, 'tta_labels.npy'),       tta_labels)
        np.save(os.path.join(OUT_DIR, 'tta_val_sum.npy'),      tta_val_sum)
        np.save(os.path.join(OUT_DIR, 'tta_val_labels.npy'),   tta_val_labels)
        np.save(os.path.join(OUT_DIR, 'tta_thresh_preds.npy'), tta_thresh_preds)
        with open(os.path.join(OUT_DIR, 'calibrated_thresholds.json'), 'w') as f:
            json.dump(calibrated_thresholds, f, indent=2)
        with open(os.path.join(OUT_DIR, 'classes_used.json'), 'w') as f:
            json.dump(classes_used, f)
        log("✓ Arrays guardados en Drive (disponibles para LOAD_TTA_FROM_DIR)")

        generate_results_pdf(
            out_dir=OUT_DIR, classes_used=classes_used, class_counts=class_counts,
            total_train=total_train, total_val=total_val, total_test=total_test,
            total_imgs=total_imgs, all_histories=all_histories, model_names=model_names,
            best_model_name=best_model_name, final_cm=final_cm, metrics_df=metrics_df,
            metrics_argmax_df=metrics_argmax_df, final_probs=np.array(final_probs),
            final_labels=np.array(final_labels), ens_bacc=ens_bacc,
            summary_df=summary_df, trained_models=trained_models,
            test_dataset=trained_test_datasets[best_model_name],
            model_times=model_times, calibrated_thresholds=calibrated_thresholds,
            tta_bacc=tta_bacc, tta_thresh_bacc=tta_thresh_bacc,
            tta_sum_probs=tta_sum_probs, tta_labels=tta_labels,
            tta_thresh_preds=tta_thresh_preds, metrics_thresh_df=metrics_thresh_df,
            mel_sens_baseline=mel_sens_baseline, mel_spec_baseline=mel_spec_baseline,
            bacc_malignas_argmax=bacc_malignas_argmax,
            bacc_malignas_thresh=bacc_malignas_thresh,
            malignant_classes=malignant_classes, tta_val_labels=tta_val_labels,
            threshold_only=False, original_thresholds=None, load_tta_source=None,
        )
        log(f"\n✓ PDF generado: {os.path.join(OUT_DIR, 'resultados_tfg.pdf')}")
        log(f"✓ Todos los outputs en: {OUT_DIR}")

        try:
            _zip_name     = os.path.basename(OUT_DIR)
            _zip_base     = os.path.join(OUTPUTS_ROOT, _zip_name)
            _zip_full     = shutil.make_archive(
                base_name = _zip_base,
                format    = 'zip',
                root_dir  = OUTPUTS_ROOT,
                base_dir  = _zip_name,
            )
            _zip_mb = os.path.getsize(_zip_full) / 1e6
            log(f"✓ ZIP creado: {_zip_full}  ({_zip_mb:.1f} MB)")
        except Exception as _ze:
            log(f"⚠ No se pudo crear el ZIP: {_ze}")

    except Exception as e:
        log(f"\n✗ ERROR {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
    except KeyboardInterrupt:
        log("\n✗ Ejecución interrumpida (KeyboardInterrupt)")
        raise
    finally:
        sys.stdout = tee.terminal
        tee.close()


# =============================================================================
#  16. PDF DE RESULTADOS
#  threshold_only=True → modo reducido (sin curvas, sin Grad-CAM)
#  threshold_only=False → PDF completo (entrenamiento normal)
# =============================================================================
def generate_results_pdf(
    out_dir, classes_used, class_counts, total_train, total_val, total_test,
    total_imgs, all_histories, model_names, best_model_name, final_cm, metrics_df,
    metrics_argmax_df, final_probs, final_labels, ens_bacc, summary_df,
    trained_models, test_dataset, model_times, calibrated_thresholds,
    tta_bacc, tta_thresh_bacc, tta_sum_probs, tta_labels, tta_thresh_preds,
    metrics_thresh_df, mel_sens_baseline, mel_spec_baseline,
    bacc_malignas_argmax, bacc_malignas_thresh, malignant_classes,
    tta_val_labels, threshold_only=False,
    original_thresholds=None, load_tta_source=None,
):
    """
    Genera el PDF completo de resultados para una ejecución.

    Incluye portada, ficha técnica, distribución de clases, imágenes
    representativas, curvas de entrenamiento, matrices de confusión,
    métricas clínicas, comparativa argmax vs umbrales, curvas ROC,
    Precision-Recall, tabla comparativa, tiempos y Grad-CAM.

    threshold_only=True genera un PDF reducido sin curvas de entrenamiento
    cuando se usa en modo LOAD_TTA_FROM_DIR.
    """
    pdf_path = os.path.join(out_dir, "resultados_tfg.pdf")

    with PdfPages(pdf_path) as pdf:

        # ── PORTADA ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        titulo = ("Clasificación automática de lesiones cutáneas\n"
                  "mediante redes neuronales convolucionales")
        ax.text(0.5, 0.82, titulo, ha='center', va='center',
                fontsize=16, fontweight='bold', transform=ax.transAxes)

        modo_lbl = "EXPERIMENTO DE UMBRALES  —  " if threshold_only else ""
        ax.text(0.5, 0.73, f"{modo_lbl}Semilla RNG {RNG_SEED}",
                ha='center', fontsize=10, color='gray', style='italic',
                transform=ax.transAxes)

        ax.text(0.5, 0.60,
                f"TTA Ensemble BACC: {tta_bacc:.4f}  |  "
                f"Con umbrales clínicos: {tta_thresh_bacc:.4f}",
                ha='center', fontsize=14, fontweight='bold', color='steelblue',
                transform=ax.transAxes)

        crit_lines = []
        for cls in THRESHOLD_PRIORITY:
            if cls in calibrated_thresholds:
                row = metrics_thresh_df[metrics_thresh_df['Clase'] == cls]
                if len(row):
                    crit_lines.append(
                        f"{cls}  sens={row.iloc[0]['Sensibilidad']:.3f}"
                        f"  spec={row.iloc[0]['Especificidad']:.3f}"
                    )
        ax.text(0.5, 0.50, "  |  ".join(crit_lines),
                ha='center', fontsize=11, color='darkorange',
                transform=ax.transAxes)

        ax.text(0.5, 0.41,
                f"BACC malignas (MEL+BCC+AKIEC):  "
                f"argmax {bacc_malignas_argmax:.4f}  →  "
                f"umbrales {bacc_malignas_thresh:.4f}",
                ha='center', fontsize=10, color='darkgreen',
                transform=ax.transAxes)

        ax.axhline(y=0.34, xmin=0.1, xmax=0.9, color='lightgray', linewidth=0.8)

        if not threshold_only and total_imgs:
            ax.text(0.5, 0.28,
                    f"Dataset: {total_imgs} imágenes  |  "
                    f"Train {total_train} / Val {total_val} / Test {total_test}  |  "
                    f"Clases: {', '.join(classes_used)}",
                    ha='center', fontsize=9, color='gray', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.28, f"Clases: {', '.join(classes_used)}",
                    ha='center', fontsize=9, color='gray', transform=ax.transAxes)

        if not threshold_only and model_times:
            total_time   = sum(model_times.values())
            tmins, tsecs = divmod(int(total_time), 60)
            ax.text(0.5, 0.20,
                    f"Tiempo total de entrenamiento: {tmins}m {tsecs}s",
                    ha='center', fontsize=9, color='gray', transform=ax.transAxes)

        pdf.savefig(fig); plt.close(fig)

        # ── FICHA TÉCNICA ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis('off')
        ax.set_title("Ficha técnica de la ejecución", fontsize=13,
                     fontweight='bold', pad=16)

        gamma_str = ", ".join(
            f"{n}:γ={FOCAL_GAMMA[n]}"
            for n in ['resnet50', 'densenet121', 'efficientnet_b3']
        )

        cfg_lines = [
            f"Semilla RNG : {RNG_SEED}",
            f"Split       : Train {TRAIN_RATIO*100:.0f}% / Val {VAL_RATIO*100:.0f}% / Test {TEST_RATIO*100:.0f}%",
            f"TTA rondas  : {TTA_ROUNDS}",
            f"Max epochs  : {MAX_EPOCHS}",
            "",
            "MODELOS",
        ]
        model_names_order = ['resnet50', 'densenet121', 'efficientnet_b3']
        for n in model_names_order:
            cfg_lines.append(
                f"  {n:<20s}  lr={LR[n]}  wd={WEIGHT_DECAY[n]}  "
                f"bs={BATCH_SIZE[n]}  {IMG_SIZE[n]}px  "
                f"ES pat={ES_PATIENCE[n]} δ={ES_MIN_DELTA[n]} w={ES_BACC_WEIGHTS[n]}"
            )
        cfg_lines += [
            "",
            "LOSS",
            f"  Train : FocalLoss per-model ({gamma_str})  +  label_smoothing=0.1",
            f"  Val   : CrossEntropyLoss (sin suavizado — interpretable para ES)",
            "",
            "DESBALANCE",
            f"  WeightedRandomSampler  +  Augmentation graduada (ratio 2/6/15)",
            f"  Clinical aug boost: " + "  ".join(
                f"{k}→nivel≥{v}" for k, v in CLINICAL_AUG_BOOST.items()
            ),
            "",
            "UMBRALES CLÍNICOS",
        ]
        for cls, tgt in CLINICAL_THRESHOLDS.items():
            theta = calibrated_thresholds.get(cls, "—")
            theta_str = f"{theta:.3f}" if isinstance(theta, float) else theta
            metodo = "argmax-θ" if not tgt['use_fbeta'] else f"F-beta β={tgt['beta']}"
            cfg_lines.append(
                f"  {cls:<6s}  sens≥{tgt['sensitivity_target']}  "
                f"spec≥{tgt['specificity_floor']}  [{metodo}]  "
                f"→ theta calibrado={theta_str}"
            )
        if not threshold_only and model_times and all_histories and summary_df is not None:
            cfg_lines += ["", "ENTRENAMIENTO"]
            for n in model_names_order:
                if n in model_times and all_histories and all_histories.get(n):
                    row_s   = summary_df[summary_df['Modelo'] == n]
                    ep_best = str(row_s.iloc[0]['Mejor epoch'])    if len(row_s) else "—"
                    ep_tot  = str(row_s.iloc[0]['Epochs totales']) if len(row_s) else "—"
                    val_b   = float(row_s.iloc[0]['Val BACC'])     if len(row_s) else 0.0
                    t_str   = str(row_s.iloc[0]['Tiempo'])         if len(row_s) else "—"
                    cfg_lines.append(
                        f"  {n:<20s}  val BACC={val_b:.4f}  "
                        f"epoch {ep_best}/{ep_tot}  "
                        f"tiempo={t_str}"
                    )
        elif threshold_only and load_tta_source:
            cfg_lines += [
                "",
                f"FUENTE ARRAYS : {os.path.basename(load_tta_source)}",
            ]
            if original_thresholds:
                cfg_lines.append(
                    "Umbrales originales: " + "  ".join(
                        f"{k}={v:.3f}" for k, v in original_thresholds.items()
                    )
                )

        ax.text(0.04, 0.95, "\n".join(cfg_lines),
                transform=ax.transAxes, fontsize=8.5,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f8f8f8', alpha=0.7))
        pdf.savefig(fig); plt.close(fig)

        # ── DISTRIBUCIÓN DE CLASES ────────────────────────────────────────
        if not threshold_only and class_counts:
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(classes_used)); w = 0.25
            tr_c = [class_counts[c][0] for c in classes_used]
            va_c = [class_counts[c][1] for c in classes_used]
            te_c = [class_counts[c][2] for c in classes_used]
            b1 = ax.bar(x - w, tr_c, w, label='Train', color='steelblue')
            b2 = ax.bar(x,     va_c, w, label='Val',   color='coral')
            b3 = ax.bar(x + w, te_c, w, label='Test',  color='mediumseagreen')
            for bar in list(b1) + list(b2) + list(b3):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        int(bar.get_height()), ha='center', va='bottom', fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(classes_used)
            ax.set_ylabel('Número de imágenes')
            ax.set_title('Distribución de imágenes por clase y split')
            ax.legend(); ax.grid(alpha=0.3)
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── IMÁGENES REPRESENTATIVAS ──────────────────────────────────────
        if not threshold_only and test_dataset is not None:
            try:
                _sample_paths = {}
                for _path, _tc in test_dataset.samples:
                    if _tc not in _sample_paths:
                        _sample_paths[_tc] = _path
                    if len(_sample_paths) == len(classes_used):
                        break

                _ncols = len(classes_used)
                fig, axes = plt.subplots(1, _ncols, figsize=(_ncols * 2.4, 3.6))
                fig.suptitle(
                    "Muestras representativas del conjunto de test  —  una imagen por clase",
                    fontsize=11, fontweight='bold', y=1.02
                )
                _cls_short = {
                    'MEL':   'Melanoma',
                    'NV':    'Nevus melanocitico',
                    'BCC':   'Carcinoma basocelular',
                    'AKIEC': 'Queratosis actinica',
                    'BKL':   'Queratosis benigna',
                    'DF':    'Dermatofibroma',
                    'VASC':  'Lesion vascular',
                }
                _malignant = {'MEL', 'BCC', 'AKIEC'}
                for _col, _cls in enumerate(classes_used):
                    _ax = axes[_col]
                    _path = _sample_paths.get(_cls)
                    if _path:
                        _img = np.array(Image.open(_path).convert("RGB")
                                        .resize((224, 224)))
                        _ax.imshow(_img)
                    _ax.axis('off')
                    _color = '#c0392b' if _cls in _malignant else '#2c3e50'
                    _ax.set_title(
                        f"{_cls}  {_cls_short.get(_cls, _cls)}",
                        fontsize=8, color=_color, fontweight='bold', pad=4
                    )
                    for _spine in _ax.spines.values():
                        _spine.set_visible(True)
                        _spine.set_edgecolor('#c0392b' if _cls in _malignant
                                             else '#bdc3c7')
                        _spine.set_linewidth(2)
                fig.text(0.02, -0.04,
                         "Rojo = clases malignas/premalignas  |  "
                         "Imágenes dermoscópicas reales del conjunto de test",
                         fontsize=8, color='gray', style='italic')
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            except Exception as _e:
                log(f"⚠ Imágenes representativas omitidas: {_e}")

        # ── CURVAS DE ENTRENAMIENTO ───────────────────────────────────────
        if not threshold_only and all_histories and model_names:
            for name in model_names:
                hist = all_histories.get(name)
                if hist is None:
                    fig, ax = plt.subplots(figsize=(8, 3)); ax.axis('off')
                    ax.text(0.5, 0.5,
                            f"Curvas de {name} no disponibles\n"
                            f"(checkpoint cargado sin _history.json)",
                            ha='center', va='center', fontsize=12, color='gray',
                            transform=ax.transAxes)
                    pdf.savefig(fig); plt.close(fig); continue

                epochs = range(1, len(hist['train_bacc']) + 1)
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(f"Curvas de entrenamiento — {name}", fontsize=14)
                axes[0].plot(epochs, hist['train_loss'], label='Train Loss', marker='o', ms=3)
                axes[0].plot(epochs, hist['val_loss'],   label='Val Loss',   marker='s', ms=3, ls='--')
                axes[0].set_title(
                    f"Pérdida — Train: FocalLoss γ={FOCAL_GAMMA[name]} | "
                    f"Val: CrossEntropyLoss\n(escalas no comparables — solo tendencia)")
                axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Pérdida")
                axes[0].legend(); axes[0].grid(alpha=0.3)
                axes[1].plot(epochs, hist['train_bacc'], label='Train BACC', marker='o', ms=3)
                axes[1].plot(epochs, hist['val_bacc'],   label='Val BACC',   marker='s', ms=3, ls='--')
                axes[1].set_title("Balanced Accuracy — monitorización sobre Val")
                axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("BACC")
                axes[1].set_ylim(0, 1)
                axes[1].legend(); axes[1].grid(alpha=0.3)
                plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── MATRICES DE CONFUSIÓN ─────────────────────────────────────────
        tta_argmax_preds = tta_sum_probs.argmax(axis=1)
        cms_to_plot = []
        if not threshold_only and final_cm is not None and best_model_name:
            cms_to_plot.append((
                final_cm,
                f"{best_model_name} argmax (Referencia base)"
            ))
        cms_to_plot += [
            (confusion_matrix(tta_labels, tta_argmax_preds),
             f"TTA Ensemble argmax  (BACC={tta_bacc:.4f})"),
            (confusion_matrix(tta_labels, tta_thresh_preds),
             "TTA Ensemble + Umbrales clínicos\n"
             + "  ".join(f"θ_{c}={calibrated_thresholds[c]:.3f}"
                         for c in THRESHOLD_PRIORITY if c in calibrated_thresholds)
             + f"\nBACC={tta_thresh_bacc:.4f}  |  BACC malignas={bacc_malignas_thresh:.4f}"),
        ]
        for cm_data, title in cms_to_plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes_used, yticklabels=classes_used, ax=ax)
            ax.set_title(f"Matriz de Confusión — {title}")
            ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── TABLAS DE MÉTRICAS CLÍNICAS ───────────────────────────────────
        tables_to_plot = []
        if not threshold_only and metrics_df is not None and best_model_name:
            tables_to_plot.append((
                metrics_df,
                f"{best_model_name} argmax (referencia base)"
            ))
        tables_to_plot += [
            (metrics_argmax_df,
             f"TTA Ensemble argmax  (BACC={tta_bacc:.4f})"),
            (metrics_thresh_df,
             f"Sistema clínico real — TTA + Umbrales\n"
             f"BACC global={tta_thresh_bacc:.4f}  |  "
             f"BACC malignas={bacc_malignas_thresh:.4f}"),
        ]
        for df_m, title in tables_to_plot:
            fig, ax = plt.subplots(figsize=(10, 4)); ax.axis('off')
            ax.set_title(f"Métricas Clínicas — {title}", fontsize=11, pad=20)
            tbl = ax.table(cellText=df_m.values, colLabels=df_m.columns,
                           cellLoc='center', loc='center')
            tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
            if 'Sistema clínico' in title or 'Umbrales' in title:
                for ri, cls in enumerate(classes_used):
                    if cls in CLINICAL_THRESHOLDS:
                        for ci in range(len(df_m.columns)):
                            tbl[ri + 1, ci].set_facecolor('#fffacd')
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── COMPARATIVA CLÍNICA ───────────────────────────────────────────
        crit_classes = [c for c in THRESHOLD_PRIORITY
                        if c in calibrated_thresholds and c in classes_used]
        n_crit = len(crit_classes)
        fig, axes = plt.subplots(1, 2, figsize=(6 + 3 * n_crit, 5))
        fig.suptitle("Comparativa clínica — argmax vs umbrales calibrados (Test)",
                     fontsize=13, fontweight='bold')
        x_pos = np.arange(n_crit); bw = 0.32
        argmax_sens, umbral_sens = [], []
        ref_df = metrics_df if (not threshold_only and metrics_df is not None) \
                 else metrics_argmax_df
        for cls in crit_classes:
            row_ind   = ref_df[ref_df['Clase'] == cls]
            base_s    = float(row_ind['Sensibilidad'].iloc[0]) if len(row_ind) else 0.0
            row_thresh = metrics_thresh_df[metrics_thresh_df['Clase'] == cls]
            thr_s     = float(row_thresh['Sensibilidad'].iloc[0]) if len(row_thresh) else 0.0
            argmax_sens.append(base_s); umbral_sens.append(thr_s)
        ref_label = ("Argmax — mejor modelo individual"
                     if (not threshold_only and metrics_df is not None)
                     else "Argmax — TTA Ensemble")
        bars1 = axes[0].bar(x_pos - bw/2, argmax_sens, bw, label=ref_label,
                            color='steelblue', alpha=0.85)
        bars2 = axes[0].bar(x_pos + bw/2, umbral_sens, bw,
                            label='Con umbrales clínicos',
                            color='darkorange', alpha=0.85)
        for bar, val in list(zip(bars1, argmax_sens)) + list(zip(bars2, umbral_sens)):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom',
                         fontsize=9, fontweight='bold')
        for i, cls in enumerate(crit_classes):
            tgt = CLINICAL_THRESHOLDS[cls]['sensitivity_target']
            axes[0].axhline(y=tgt, xmin=i/n_crit, xmax=(i+1)/n_crit,
                            color='red', linestyle='--', linewidth=1.2, alpha=0.7)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(
            [f'{c}\nθ={calibrated_thresholds[c]:.3f}' for c in crit_classes]
        )
        axes[0].set_ylim(0, 1.12); axes[0].set_ylabel('Sensibilidad')
        axes[0].set_title('Sensibilidad clases críticas (rojo = objetivo)')
        axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=0.3)

        estrategias = ['TTA Ensemble\n(argmax)', 'TTA Ensemble\n(umbrales clínicos)']
        baccs       = [tta_bacc, tta_thresh_bacc]
        bars3 = axes[1].bar(estrategias, baccs,
                            color=['steelblue', 'darkorange'], alpha=0.85, width=0.4)
        for bar, val in zip(bars3, baccs):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.003, f'{val:.4f}',
                         ha='center', va='bottom', fontsize=11, fontweight='bold')
        axes[1].set_ylim(max(0, min(baccs) - 0.05), min(1, max(baccs) + 0.05))
        axes[1].set_ylabel('Test BACC'); axes[1].set_title('BACC global — TTA Ensemble')
        axes[1].grid(axis='y', alpha=0.3)
        delta = tta_thresh_bacc - tta_bacc
        axes[1].text(0.5, 0.10, f'Delta BACC global = {delta:+.4f}',
                     ha='center', transform=axes[1].transAxes, fontsize=9,
                     color='green' if delta >= 0 else 'red', fontweight='bold')
        axes[1].text(0.5, 0.03,
                     f'BACC malignas: {bacc_malignas_argmax:.4f} → '
                     f'{bacc_malignas_thresh:.4f} '
                     f'({bacc_malignas_thresh - bacc_malignas_argmax:+.4f})',
                     ha='center', transform=axes[1].transAxes,
                     fontsize=9, color='darkgreen', fontweight='bold')
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── FIABILIDAD DE UMBRALES ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5)); ax.axis('off')
        ax.set_title("Fiabilidad de los umbrales clínicos calibrados",
                     fontsize=13, pad=20)
        lines = [
            "Los umbrales se calibran sobre val y se evalúan en test.",
            "La fiabilidad depende del número de muestras val por clase.\n",
        ]
        for cls in THRESHOLD_PRIORITY:
            if cls not in calibrated_thresholds: continue
            theta   = calibrated_thresholds[cls]
            cls_idx = classes_used.index(cls)
            n_val   = int((tta_val_labels == cls_idx).sum())
            row_val = metrics_thresh_df[metrics_thresh_df['Clase'] == cls]
            s_test  = float(row_val['Sensibilidad'].iloc[0]) if len(row_val) else 0.0
            tgt     = CLINICAL_THRESHOLDS[cls]
            metodo  = f"F-beta(β={tgt['beta']})" if tgt['use_fbeta'] else "argmax-θ"
            reliab  = "ALTA (>200 muestras)" if n_val > 200 else "LIMITADA (<100 muestras)"
            ok      = ("OBJETIVO ALCANZADO"
                       if s_test >= tgt['sensitivity_target']
                       else "OBJETIVO NO ALCANZADO en test")
            lines.append(
                f"{cls}:  theta={theta:.3f}  |  método={metodo}  |  "
                f"Muestras val={n_val}  |  Fiabilidad: {reliab}"
            )
            lines.append(
                f"       Sens test={s_test:.3f}  "
                f"(objetivo >{tgt['sensitivity_target']})  — {ok}"
            )
            if n_val < 100:
                lines.append(
                    f"       NOTA: con {n_val} muestras, "
                    f"1 imagen = ±{1/n_val:.3f} de sensibilidad."
                )
            lines.append("")
        lines.append("MEL: argmax-theta con spec>={:.2f} — equilibrio sens/spec verificado.".format(
            CLINICAL_THRESHOLDS['MEL']['specificity_floor']))
        lines.append("AKIEC: argmax-θ (<100 muestras, mayor incertidumbre estadística).")
        lines.append("Para mayor robustez de AKIEC: datos adicionales (ISIC 2019/2020).")
        ax.text(0.05, 0.95, "\n".join(lines),
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── CURVAS ROC ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        y_bin = label_binarize(tta_labels, classes=range(len(classes_used)))
        for i, cls in enumerate(classes_used):
            fpr, tpr, _ = roc_curve(y_bin[:, i], tta_sum_probs[:, i])
            ax.plot(fpr, tpr, lw=2, label=f'{cls} (AUC={auc(fpr, tpr):.2f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title("Curvas ROC — TTA Ensemble (Test)")
        ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.3)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── CURVAS PRECISION-RECALL ───────────────────────────────────────
        # Complementan las ROC: más informativas con clases desbalanceadas.
        # El punto marcado (círculo) indica el umbral clínico calibrado.
        fig, ax = plt.subplots(figsize=(8, 6))
        _y_bin_pr = label_binarize(tta_labels, classes=range(len(classes_used)))
        _colors_pr = plt.cm.tab10(np.linspace(0, 0.9, len(classes_used)))
        for _i, _cls_pr in enumerate(classes_used):
            _prec_pr, _rec_pr, _ = precision_recall_curve(
                _y_bin_pr[:, _i], tta_sum_probs[:, _i])
            _ap = average_precision_score(_y_bin_pr[:, _i], tta_sum_probs[:, _i])
            ax.plot(_rec_pr, _prec_pr, lw=2,
                    label=f'{_cls_pr} (AP={_ap:.2f})',
                    color=_colors_pr[_i])
            if _cls_pr in calibrated_thresholds:
                _th_pr  = calibrated_thresholds[_cls_pr]
                _tta_sp = tta_sum_probs[:, _i]
                _pd_pr  = _tta_sp >= _th_pr
                _lb_pr  = _y_bin_pr[:, _i]
                _tp_pr  = ((_pd_pr) & (_lb_pr == 1)).sum()
                _fp_pr  = ((_pd_pr) & (_lb_pr == 0)).sum()
                _fn_pr  = ((~_pd_pr) & (_lb_pr == 1)).sum()
                _pr_op  = _tp_pr / max(_tp_pr + _fp_pr, 1)
                _rc_op  = _tp_pr / max(_tp_pr + _fn_pr, 1)
                ax.scatter(_rc_op, _pr_op, s=90, color=_colors_pr[_i],
                           zorder=5, edgecolors='black', linewidth=0.8)
        ax.set_xlabel('Recall  (Sensibilidad)')
        ax.set_ylabel('Precision  (Valor Predictivo Positivo)')
        ax.set_title(
            "Curvas Precision-Recall  —  TTA Ensemble (Test)\n"
            "Circulo = punto de operacion con umbral clinico calibrado",
            fontsize=11)
        ax.legend(fontsize=9, loc='lower left',
                  framealpha=0.9, edgecolor='lightgray')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── TABLA COMPARATIVA FINAL ───────────────────────────────────────
        if not threshold_only and summary_df is not None:
            sd = summary_df.copy()
            extra = pd.DataFrame([
                {k: "─"*9 for k in summary_df.columns},
                {"Modelo": "BACC malignas (argmax)", "Val BACC": "-",
                 "Test BACC": f"{bacc_malignas_argmax:.4f}", "Mejor epoch": "-",
                 "Epochs totales": "MEL+BCC+AKIEC", "Tiempo": "-"},
                {"Modelo": "BACC malignas (umbral)", "Val BACC": "-",
                 "Test BACC": f"{bacc_malignas_thresh:.4f}", "Mejor epoch": "-",
                 "Epochs totales": "MEL+BCC+AKIEC", "Tiempo": "-"},
            ])
            sd = pd.concat([sd, extra], ignore_index=True)
            fig, ax = plt.subplots(figsize=(11, 1.2 + len(sd) * 0.6))
            ax.axis('off')
            ax.set_title("Comparativa final de modelos", fontsize=13, pad=20)
            tbl = ax.table(cellText=sd.values, colLabels=sd.columns,
                           cellLoc='center', loc='center')
            tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.8)
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── GRÁFICA DE BARRAS ─────────────────────────────────────────────
        if not threshold_only and summary_df is not None:
            nombres  = summary_df['Modelo'].tolist()
            test_bac = [float(v) if str(v) != '-' else 0.0
                        for v in summary_df['Test BACC']]
            color_map = {
                'resnet50': 'steelblue', 'densenet121': 'steelblue',
                'efficientnet_b3': 'steelblue', 'Ensemble': 'coral',
                'TTA Ensemble': 'mediumseagreen',
                'TTA + umbrales clinicos': 'darkorange',
            }
            colores = [color_map.get(n, 'mediumseagreen') for n in nombres]
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            fig.suptitle('Comparativa de estrategias de inferencia',
                         fontsize=13, fontweight='bold')
            bars1 = axes[0].bar(nombres, test_bac, color=colores,
                                edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars1, test_bac):
                if val > 0:
                    axes[0].text(bar.get_x() + bar.get_width()/2,
                                 bar.get_height() + 0.002, f'{val:.4f}',
                                 ha='center', va='bottom', fontsize=8)
            axes[0].set_ylim(0, 1); axes[0].set_ylabel('Test BACC global')
            axes[0].set_title('BACC global (métrica oficial ISIC)')
            axes[0].tick_params(axis='x', rotation=35)
            plt.setp(axes[0].get_xticklabels(), ha='right')
            axes[0].grid(axis='y', alpha=0.3)
            if test_bac:
                axes[0].axhline(y=max(test_bac), color='red', linestyle='--',
                                linewidth=0.8, alpha=0.5,
                                label=f'Mejor: {max(test_bac):.4f}')
                axes[0].legend(fontsize=8)
            nombres_m   = ['TTA Ensemble', 'TTA + umbrales clínicos']
            bacc_m_vals = [bacc_malignas_argmax, bacc_malignas_thresh]
            bars2 = axes[1].bar(nombres_m, bacc_m_vals,
                                color=['mediumseagreen', 'darkorange'],
                                edgecolor='white', linewidth=0.5, width=0.4)
            for bar, val in zip(bars2, bacc_m_vals):
                axes[1].text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.003, f'{val:.4f}',
                             ha='center', va='bottom', fontsize=11, fontweight='bold')
            delta_m = bacc_malignas_thresh - bacc_malignas_argmax
            axes[1].set_ylim(max(0, min(bacc_m_vals) - 0.05),
                             min(1, max(bacc_m_vals) + 0.08))
            axes[1].set_ylabel('BACC malignas')
            axes[1].set_title(
                f'BACC malignas MEL+BCC+AKIEC\n(métrica clínica — Delta: {delta_m:+.4f})'
            )
            axes[1].grid(axis='y', alpha=0.3)
            axes[1].text(0.5, 0.05,
                         'Los umbrales clínicos mejoran la detección\n'
                         'de lesiones malignas/premalignas',
                         ha='center', transform=axes[1].transAxes,
                         fontsize=9, color='darkgreen', style='italic')
            plt.tight_layout(rect=[0, 0.08, 1, 0.95])
            pdf.savefig(fig); plt.close(fig)

        # ── TIEMPOS DE ENTRENAMIENTO ──────────────────────────────────────
        if not threshold_only and model_times:
            fig, ax = plt.subplots(figsize=(8, 4))
            nt = list(model_times.keys())
            tm = [model_times[n] / 60 for n in nt]
            bars = ax.bar(nt, tm, color=['steelblue', 'coral', 'mediumseagreen'])
            for bar, t in zip(bars, tm):
                mb_, sb_ = divmod(int(t * 60), 60)
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3, f"{mb_}m {sb_}s",
                        ha='center', va='bottom', fontsize=10)
            ax.set_ylabel("Tiempo (minutos)")
            ax.set_title("Tiempo de entrenamiento por modelo")
            ax.set_ylim(0, max(tm) * 1.18)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── GRAD-CAM ──────────────────────────────────────────────────────
        if (not threshold_only and trained_models and best_model_name
                and test_dataset is not None):
            try:
                bme = trained_models[best_model_name]
                bme.eval()
                tmp_loader = DataLoader(
                    test_dataset, batch_size=BATCH_SIZE[best_model_name],
                    shuffle=False, num_workers=NUM_WORKERS, pin_memory=False
                )
                all_tp = []
                with torch.no_grad():
                    for x, _ in tmp_loader:
                        all_tp.extend(bme(x.to(device)).argmax(1).cpu().numpy())

                cls_cands = defaultdict(list)
                for idx, (path, tc) in enumerate(test_dataset.samples):
                    ti = test_dataset.cls2idx[tc]
                    cls_cands[tc].append((idx, path, all_tp[idx] == ti))

                # Build sample_paths in CLASSES order, one per class.
                # Prefer a true positive; fall back to any available image.
                # classes_used is the authoritative ordered list — using it
                # directly avoids labeling bugs from CLASSES/cls2idx mismatches.
                sample_paths = []
                sample_classes = []  # track which class each sample belongs to
                for cls in classes_used:
                    cands = cls_cands.get(cls, [])
                    if not cands: continue
                    tp_c = [(i, p) for i, p, is_tp in cands if is_tp]
                    sample_paths.append(tp_c[0][1] if tp_c else cands[0][1])
                    sample_classes.append(cls)

                ns = len(sample_paths)
                fig, axes = plt.subplots(2, ns, figsize=(ns * 2.5, 5))
                fig.suptitle(
                    f"Grad-CAM — {best_model_name} (Test)\n"
                    f"Un verdadero positivo por clase — zonas de atención del modelo",
                    fontsize=11
                )
                for col, path in enumerate(sample_paths):
                    isz  = IMG_SIZE[best_model_name]
                    imat = np.array(Image.open(path).convert("RGB").resize((isz, isz)))
                    it   = test_dataset._val_transform(image=imat)['image']
                    with torch.no_grad():
                        pc = bme(it.unsqueeze(0).to(device)).argmax(1).item()
                    cam = compute_gradcam(bme, it, pc, best_model_name)
                    ov  = overlay_gradcam(imat, cam)
                    axes[0, col].imshow(imat); axes[0, col].axis('off')
                    axes[0, col].set_title(
                        sample_classes[col], fontsize=8, fontweight='bold'
                    )
                    axes[1, col].imshow(ov);   axes[1, col].axis('off')
                axes[0, 0].set_ylabel("Original", fontsize=9)
                axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)
                plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

            except Exception as e:
                log(f"⚠ Grad-CAM omitido: {e}")


# =============================================================================
#  EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt):
        # Renombrar carpeta con _e para indicar ejecución incompleta o fallida.
        # Así es fácil distinguir en Drive qué ejecuciones terminaron bien.
        # NOTA: si Colab se desconecta abruptamente, este bloque no se ejecuta.
        if _current_out_dir and os.path.exists(_current_out_dir):
            error_dir = _current_out_dir + "_e"
            try:
                os.rename(_current_out_dir, error_dir)
                print(f"\n⚠ Ejecución fallida/interrumpida — carpeta renombrada a: "
                      f"{os.path.basename(error_dir)}", flush=True)
            except Exception:
                pass

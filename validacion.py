import torch

import numpy as np
from tqdm import tqdm
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from .save_results import save_results_validation
import pandas as pd
from glob import glob
from model import Model
from sklearn.metrics import roc_auc_score


def get_transform_function(side_size, crop_size, mean, std, num_frames):
    def transform(clip_tensor):
        # clip_tensor: (T, H, W, C) → Normalizado y transformado a (T, C, H, W)

        # Reordenar a (T, C, H, W)
        clip_tensor = clip_tensor.permute(0, 3, 1, 2) / 255.0  # Escala a [0,1]

        # Resize corto lado (por frame)
        resized = []
        for frame in clip_tensor:
            c, h, w = frame.shape
            if h < w:
                new_h = side_size
                new_w = int(w * (side_size / h))
            else:
                new_w = side_size
                new_h = int(h * (side_size / w))
            frame_np = frame.numpy().transpose(1, 2, 0)  # C,H,W -> H,W,C
            frame_np = cv2.resize(frame_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_frame = torch.from_numpy(frame_np.transpose(2, 0, 1))
            resized.append(resized_frame)
        clip_tensor = torch.stack(resized)

        # Center crop
        _, _, h, w = clip_tensor.shape
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        clip_tensor = clip_tensor[:, :, top:top + crop_size, left:left + crop_size]

        # Normalizar
        mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std).view(1, 3, 1, 1)
        clip_tensor = (clip_tensor - mean_tensor) / std_tensor

        return clip_tensor  # (T, C, H, W)
    return transform

def cargar_anotaciones(video_path, anotaciones_txt):
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # sin extensión
    print(f"📁 Buscando anotaciones para: {video_name}")

    with open(anotaciones_txt, 'r') as f:
        for line in f:
            parts = [p for p in line.strip().split() if p != '']

            if len(parts) < 2:
                continue  # línea incompleta

            nombre_archivo = parts[0]
            if nombre_archivo != video_name:
                continue

            try:
                num_frames = int(parts[1])
            except ValueError:
                print(f"❌ Error leyendo número de frames para {nombre_archivo}")
                continue

            etiquetas = [0] * num_frames

            # Si no hay pares start/end, se asume video normal
            if len(parts) == 2:
                print(f"ℹ️ Video {video_name} sin anomalías explícitas. Etiquetas: todo 0")
                return etiquetas

            # Procesar pares de anotaciones (start, end)
            for i in range(2, len(parts), 2):
                try:
                    start = int(parts[i])
                    end = int(parts[i + 1])
                    for j in range(start, min(end + 1, num_frames)):
                        etiquetas[j] = 1
                except (IndexError, ValueError):
                    print(f"⚠️ Par de anotación malformado en línea: {line}")
                    continue

            print(f"✅ Anotaciones cargadas para {video_name} ({num_frames} frames)")
            return etiquetas

    print(f"❌ {video_name} no encontrado en {anotaciones_txt}")
    return []


def cargar_modelo_x3d(x3d_version: str, device, num_frames, stride):
    print(f"\n🔧 Cargando modelo X3D versión: {x3d_version} en dispositivo: {device}")
    x3d_versions_map = {
        "xs": "x3d_xs",
        "s": "x3d_s",
        "m": "x3d_m",
        "l": "x3d_l"
    }

    if x3d_version not in x3d_versions_map:
        raise ValueError(f"Versión x3d desconocida: {x3d_version}")

    model_name = x3d_versions_map[x3d_version]
    model_x3d = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    model_x3d = model_x3d.eval().to(device)
    print(f"✅ Modelo X3D {model_name} cargado.")

    if hasattr(model_x3d, "blocks"):
        print(f"ℹ️ Eliminando última capa del modelo...")
        model_x3d.blocks = model_x3d.blocks[:-1]

    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]

    model_transform_params = {
        "x3d_xs": {"side_size": 182, "crop_size": 182, "num_frames": num_frames, "sampling_rate": stride},
        "x3d_s":  {"side_size": 182, "crop_size": 182, "num_frames": num_frames, "sampling_rate": stride},
        "x3d_m":  {"side_size": 256, "crop_size": 256, "num_frames": num_frames, "sampling_rate": stride},
        "x3d_l":  {"side_size": 320, "crop_size": 320, "num_frames": num_frames, "sampling_rate": stride}
    }

    transform_params = model_transform_params[model_name]
    print(f"🌀 Transform Params: {transform_params}")


    transform = get_transform_function(
    side_size=transform_params["side_size"],
    crop_size=transform_params["crop_size"],
    mean=mean,
    std=std,
    num_frames=num_frames   
    )

    #if hasattr(model_x3d, "blocks"):
    #    print(f"ℹ️ Eliminando última capa del modelo...")
    #    model_x3d.blocks = model_x3d.blocks[:-1]

    return model_x3d, transform


def procesar_video(video_path, anotaciones, num_frames, stride, device, transform, model_x3d, model_custom, batch_size=32):
    print(f"\n🎥 Procesando video: {video_path}")
    clip_frame_count = num_frames * stride

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el video: {video_path}")
        return [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🔢 Total de frames: {total_frames}")
    frames_buffer = []
    batch_buffer = []
    predicciones_final = []

    pbar = tqdm(total=total_frames, desc=f"Procesando {os.path.basename(video_path)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_buffer.append(frame)
        pbar.update(1)

        if len(frames_buffer) == clip_frame_count:
            clip_np = np.stack(frames_buffer, axis=0).astype(np.float32)
            clip_tensor = torch.from_numpy(clip_np)  # (T, H, W, C)
            transformed_clip = transform(clip_tensor)  # (T, C, H, W)
            batch_buffer.append(transformed_clip)
            frames_buffer = []

            if len(batch_buffer) == batch_size:
                batch_tensor = torch.stack(batch_buffer).to(device)  # (B, T, C, H, W)
                with torch.no_grad():
                    features = model_x3d(batch_tensor).cpu().numpy()  # (B, F)
                feature_tensor = torch.from_numpy(features).to(device)

                with torch.no_grad():
                    scores, _ = model_custom(feature_tensor)
                    scores = torch.sigmoid(scores).squeeze().cpu().numpy()  # (B,)
                
                for score in scores:
                    predicciones_final.extend([score] * clip_frame_count)

                batch_buffer = []

    # Procesar el último batch incompleto
    if batch_buffer:
        batch_tensor = torch.stack(batch_buffer).to(device)
        with torch.no_grad():
            features = model_x3d(batch_tensor).cpu().numpy()
        feature_tensor = torch.from_numpy(features).to(device)

        with torch.no_grad():
            scores, _ = model_custom(feature_tensor)
            scores = torch.sigmoid(scores).squeeze().cpu().numpy()

        for score in scores:
            predicciones_final.extend([score] * clip_frame_count)

    cap.release()
    pbar.close()

    anotaciones = anotaciones[:len(predicciones_final)]
    print(f"📏 predicciones: {len(predicciones_final)}, etiquetas: {len(anotaciones)}")
    return predicciones_final, anotaciones


def procesar_carpeta(input_folder, anotaciones_txt, num_frames, stride, x3d_model, model_custom_name, arch):
    import os  # por si no está importado

    print(f"\n📂 Procesando carpeta: {input_folder}")
    video_files = glob(os.path.join(input_folder, "*.*"))
    video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    device = torch.device("cpu")

    all_scores = []
    all_labels = []

    s_scores = []
    s_labels = []

    m_scores =[]
    m_labels =[]

    l_scores =[]
    l_labels =[]

    # Crear carpeta resultados si no existe
    os.makedirs("./resultados", exist_ok=True)

    # Modelo anomalía
    model_path = f'./ckpt/{model_custom_name}.pkl'
    print(f"\n📦 Cargando modelo de anomalía desde: {model_path}")
    if arch == 'base':
        model_custom = Model()
    elif arch in ['fast', 'tiny']:
        model_custom = Model(ff_mult=1, dims=(32, 32), depths=(1, 1))
    else:
        raise ValueError("Arquitectura desconocida.")

    model_custom.load_state_dict(torch.load(model_path, map_location=device))
    model_custom.eval().to(device)

    # Modelo X3D
    model_x3d, transform = cargar_modelo_x3d(x3d_model, device, num_frames, stride)

    with open("list/validacion_l.txt", "r", encoding="utf-8") as f:
        validacion_l = [line.strip() for line in f]

    with open("list/validacion_m.txt", "r", encoding="utf-8") as f:
        validacion_m = [line.strip() for line in f]

    with open("list/validacion_s.txt", "r", encoding="utf-8") as f:
        validacion_s = [line.strip() for line in f]

    for video_path in video_files:
        anotaciones = cargar_anotaciones(video_path, anotaciones_txt)
        nombre_base = os.path.splitext(os.path.basename(video_path))[0]
        if len(anotaciones) == 0:
            print(f"⚠️ No hay anotaciones para {os.path.basename(video_path)}. Se omite.")
            continue

        predicciones, etiquetas = procesar_video(
            video_path, anotaciones, num_frames, stride, device, transform, model_x3d, model_custom
        )

        all_scores.extend(predicciones)
        all_labels.extend(etiquetas)

        if nombre_base in validacion_l:
            l_scores.extend(predicciones)
            l_labels.extend(etiquetas)

        if nombre_base in validacion_m:
            m_scores.extend(predicciones)
            m_labels.extend(etiquetas)

        if nombre_base in validacion_s:
            s_scores.extend(predicciones)
            s_labels.extend(etiquetas)


    
    auc = roc_auc_score(all_labels, all_scores)
    print(f"\n📈 AUC ROC global: {auc:.4f}")

    auc_l = roc_auc_score(l_labels, l_scores)
    print(f"\n📈 AUC ROC s: {auc_l:.4f}")

    auc_m = roc_auc_score(m_labels, m_scores)
    print(f"\n📈 AUC ROC m: {auc_m:.4f}")

    auc_s = roc_auc_score(s_labels, s_scores)
    print(f"\n📈 AUC ROC l: {auc_s:.4f}")

    save_results_validation(model_custom_name, auc, auc_l, auc_m, auc_s)

    print(f"✅ Resultados guardados para {model_custom_name}")

    

import re

import os
import re

def evaluar_todos_los_modelos(carpeta_ckpt="./ckpt"):
    modelos = [f for f in os.listdir(carpeta_ckpt) if f.endswith(".pkl") and f.startswith("STEAD_")]

    # 🔀 Ordenar: STEAD_FAST* primero
    modelos = sorted(modelos, key=lambda x: 0 if x.startswith("STEAD_FAST") else 1)

    patron = r"STEAD_(\w+)_(\w+)_(\d+)_(\d+)final"

    anotaciones_txt = "./VALIDACION/annotations.txt"
    input_folder = "VALIDACION"
    carpeta_resultados = "./validacion_resultados"  # puedes ajustarlo si lo guardas en otro lugar

    for modelo in modelos:
        nombre = os.path.splitext(modelo)[0]  # sin .pkl
        coincidencia = re.match(patron, nombre)

        if not coincidencia:
            print(f"❌ Nombre no válido, se omite: {modelo}")
            continue

        arch, x3d_model, num_frames, stride = coincidencia.groups()
        arch = arch.lower()
        x3d_model = x3d_model.lower()
        num_frames = int(num_frames)
        stride = int(stride)

        # 🧠 Verificar si ya fue procesado (puede cambiarse a .txt, .json, etc. según tu output)
        nombre_csv = f"{nombre}"
        path_csv = os.path.join(carpeta_resultados, nombre_csv)

        if os.path.exists(path_csv):
            print(f"✅ Modelo ya procesado: {nombre}. Se omite.")
            continue

        print(f"\n🚀 Evaluando modelo: {nombre}")
        procesar_carpeta(
            input_folder=input_folder,
            anotaciones_txt=anotaciones_txt,
            num_frames=num_frames,
            stride=stride,
            x3d_model=x3d_model,
            model_custom_name=nombre,
            arch=arch
        )

# === USO ===
if __name__ == "__main__":
    evaluar_todos_los_modelos()



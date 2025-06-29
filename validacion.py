import torch
import numpy as np
import cv2
import os
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from save_results import save_results_validation
from model import Model

def get_transform_function(side_size, crop_size, mean, std, num_frames):
    def transform(clip_tensor):
        clip_tensor = clip_tensor.permute(0, 3, 1, 2) / 255.0
        resized = []
        for frame in clip_tensor:
            c, h, w = frame.shape
            if h < w:
                new_h = side_size
                new_w = int(w * (side_size / h))
            else:
                new_w = side_size
                new_h = int(h * (side_size / w))
            frame_np = frame.numpy().transpose(1, 2, 0)
            frame_np = cv2.resize(frame_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_frame = torch.from_numpy(frame_np.transpose(2, 0, 1))
            resized.append(resized_frame)
        clip_tensor = torch.stack(resized)
        _, _, h, w = clip_tensor.shape
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        clip_tensor = clip_tensor[:, :, top:top + crop_size, left:left + crop_size]
        mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std).view(1, 3, 1, 1)
        clip_tensor = (clip_tensor - mean_tensor) / std_tensor
        clip_tensor = clip_tensor.permute(1, 0, 2, 3)
        return clip_tensor
    return transform

def cargar_anotaciones(video_path, anotaciones_txt):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    with open(anotaciones_txt, 'r') as f:
        for line in f:
            parts = [p for p in line.strip().split() if p != '']
            if len(parts) < 2:
                continue
            nombre_archivo = parts[0]
            if nombre_archivo != video_name:
                continue
            try:
                num_frames = int(parts[1])
            except ValueError:
                continue
            etiquetas = [0] * num_frames
            if len(parts) == 2:
                return etiquetas
            for i in range(2, len(parts), 2):
                try:
                    start = int(parts[i])
                    end = int(parts[i + 1])
                    for j in range(start, min(end + 1, num_frames)):
                        etiquetas[j] = 1
                except (IndexError, ValueError):
                    continue
            return etiquetas
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
    if hasattr(model_x3d, "blocks"):
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
    transform = get_transform_function(
        side_size=transform_params["side_size"],
        crop_size=transform_params["crop_size"],
        mean=mean,
        std=std,
        num_frames=num_frames
    )
    return model_x3d, transform

def procesar_video(video_path, anotaciones, num_frames, stride, device, transform, model_x3d, model_custom, batch_size=128):
    clip_frame_count = num_frames * stride
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_buffer = []
    batch_buffer = []
    predicciones_final = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_buffer.append(frame)
        if len(frames_buffer) == clip_frame_count:
            clip_np = np.stack(frames_buffer, axis=0).astype(np.float32)
            clip_tensor = torch.from_numpy(clip_np)
            transformed_clip = transform(clip_tensor)
            batch_buffer.append(transformed_clip)
            frames_buffer = []
            if len(batch_buffer) == batch_size:
                batch_tensor = torch.stack(batch_buffer).to(device)
                with torch.no_grad():
                    features = model_x3d(batch_tensor).cpu().numpy()
                feature_tensor = torch.from_numpy(features).to(device)
                with torch.no_grad():
                    scores, _ = model_custom(feature_tensor)
                    scores = torch.sigmoid(scores).squeeze().cpu().numpy()
                for score in scores:
                    predicciones_final.extend([score] * clip_frame_count)
                batch_buffer = []
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
    anotaciones = anotaciones[:len(predicciones_final)]
    return predicciones_final, anotaciones

def procesar_carpeta(input_folder, anotaciones_txt, num_frames, stride, x3d_model, model_custom_name, arch):
    print(f"\n📂 Procesando carpeta: {input_folder}")
    video_files = glob(os.path.join(input_folder, "*.*"))
    video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}")
    all_scores, all_labels = [], []
    s_scores, s_labels, m_scores, m_labels, l_scores, l_labels = [], [], [], [], [], []
    os.makedirs("./resultados", exist_ok=True)
    model_path = f'./ckpt/{model_custom_name}.pkl'
    if arch == 'base':
        model_custom = Model()
    elif arch in ['fast', 'tiny']:
        model_custom = Model(ff_mult=1, dims=(32, 32), depths=(1, 1))
    else:
        raise ValueError("Arquitectura desconocida.")
    model_custom.load_state_dict(torch.load(model_path, map_location=device))
    model_custom.eval().to(device)
    model_x3d, transform = cargar_modelo_x3d(x3d_model, device, num_frames, stride)
    with open("list/validacion_l.txt", "r") as f:
        validacion_l = [line.strip() for line in f]
    with open("list/validacion_m.txt", "r") as f:
        validacion_m = [line.strip() for line in f]
    with open("list/validacion_s.txt", "r") as f:
        validacion_s = [line.strip() for line in f]
    video_bar = tqdm(video_files, desc="Procesando videos", unit="video")
    for video_path in video_bar:
        anotaciones = cargar_anotaciones(video_path, anotaciones_txt)
        nombre_base = os.path.splitext(os.path.basename(video_path))[0]
        if len(anotaciones) == 0:
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
    auc_l = roc_auc_score(l_labels, l_scores) if l_labels else 0
    auc_m = roc_auc_score(m_labels, m_scores) if m_labels else 0
    auc_s = roc_auc_score(s_labels, s_scores) if s_labels else 0
    print(f"\n📈 AUC ROC global: {auc:.4f}")
    print(f"📈 AUC ROC s: {auc_s:.4f}")
    print(f"📈 AUC ROC m: {auc_m:.4f}")
    print(f"📈 AUC ROC l: {auc_l:.4f}")
    save_results_validation(model_custom_name, auc, auc_l, auc_m, auc_s)
    print(f"✅ Resultados guardados para {model_custom_name}")

def evaluar_todos_los_modelos(carpeta_ckpt="./ckpt"):
    modelos = [f for f in os.listdir(carpeta_ckpt) if f.endswith(".pkl") and f.startswith("STEAD_")]
    modelos = sorted(modelos, key=lambda x: 0 if x.startswith("STEAD_FAST") else 1)
    patron = r"STEAD_(\w+)_(\w+)_(\d+)_(\d+)final"
    anotaciones_txt = "./VALIDACION/annotations.txt"
    input_folder = "VALIDACION"
    carpeta_resultados = "./validacion_resultados"
    for modelo in modelos:
        nombre = os.path.splitext(modelo)[0]
        coincidencia = re.match(patron, nombre)
        if not coincidencia:
            continue
        arch, x3d_model, num_frames, stride = coincidencia.groups()
        arch = arch.lower()
        x3d_model = x3d_model.lower()
        num_frames = int(num_frames)
        stride = int(stride)
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

if __name__ == "__main__":
    evaluar_todos_los_modelos()

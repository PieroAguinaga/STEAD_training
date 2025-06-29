import os
import cv2
import torch
import numpy as np
from torchvision.transforms import Resize, CenterCrop, Normalize
from tqdm import tqdm

# Configuración de parámetros por modelo

# Normalización estándar (imagen en rango [0,1])
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]


def extract_features_folder(
    input_folder: str,
    output_folder: str,
    model_name: str,
    sampling_rate: str,
    device: str = 'cpu',
    batch_size: int = 1,
    num_frames: int = 1,
):
    """
    Extrae features de videos en input_folder usando modelo X3D sin pytorchvideo transforms.

    Args:
        input_folder (str): carpeta con videos .mp4
        output_folder (str): carpeta destino donde guardar .npy
        model_name (str): uno de ['x3d_xs','x3d_s','x3d_m','x3d_l']
        device (str): 'cpu' o 'cuda'
        batch_size (int): cantidad de videos a procesar en batch para inferencia
    """
    MODEL_PARAMS = {
    'x3d_xs': {'side_size': 182, 'crop_size': 182, 'num_frames': num_frames},#4
    'x3d_s':  {'side_size': 182, 'crop_size': 182, 'num_frames': num_frames},#13
    'x3d_m':  {'side_size': 256, 'crop_size': 256, 'num_frames': num_frames},#16
    'x3d_l':  {'side_size': 320, 'crop_size': 320, 'num_frames': num_frames},

}
    os.makedirs(output_folder, exist_ok=True)
    assert model_name in MODEL_PARAMS, f"Modelo no soportado: {model_name}"
    params = MODEL_PARAMS[model_name]

    # Configurar dispositivo
    device = torch.device(device)
    print(f"🔧 Usando dispositivo: {device}")

    # Cargar modelo X3D pre-entrenado y quitar última capa de clasificación
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    model.eval().to(device)
    if hasattr(model, 'blocks'):
        del model.blocks[-1]

    # Transforms: Resize, CenterCrop, Normalize
    resize = Resize(params['side_size'])
    centercrop = CenterCrop((params['crop_size'], params['crop_size']))
    normalize = Normalize(MEAN, STD)

    # Preparar lista de archivos
    files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

    # Buffer para batches
    batch_buffer = []

    for fname in tqdm(files, desc='Videos procesando'):
        base, _ = os.path.splitext(fname)
        out_path = os.path.join(output_folder, base + '.npy')
        if os.path.exists(out_path):
            continue  # ya procesado

        video_path = os.path.join(input_folder, fname)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                # BGR -> RGB y float
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        cap.release()

        total_frames = len(frames)
        if total_frames < params['num_frames']:
            print(f"⚠️ Video muy corto: {fname}")
            continue

        # Muestreo uniforme de frames
        
        sampled_frames = frames[::sampling_rate]

        # Selecciona los primeros num_frames (rellena si hay pocos)
        if len(sampled_frames) < params['num_frames']:
            print(f"⚠️ Clip corto tras submuestreo: {fname}")
            # Rellenar repitiendo el último frame
            pad_frame = sampled_frames[-1]
            pad = [pad_frame] * (params['num_frames'] - len(sampled_frames))
            sampled_frames += pad
        else:
            sampled_frames = sampled_frames[:params['num_frames']]
        clip = sampled_frames

        # Transformar clip a tensor
        torch_clip = []
        for img in clip:
            tensor = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W
            tensor = resize(tensor)
            tensor = centercrop(tensor)
            tensor = normalize(tensor)
            torch_clip.append(tensor)

        # Generar tensor de batch si corresponde
        video_tensor = torch.stack(torch_clip, dim=1).unsqueeze(0).to(device)  # (1,C,T,H,W)
        batch_buffer.append((video_tensor, out_path))

        # Si el buffer alcanzó el tamaño batch_size, procesar
        if len(batch_buffer) >= batch_size:
            _process_batch(batch_buffer, model, device)
            batch_buffer = []

    # Procesar resto de buffer
    if batch_buffer:
        _process_batch(batch_buffer, model, device)


def _process_batch(batch_buffer, model, device):
    """Realiza la inferencia del batch y guarda los .npy."""
    tensors = torch.cat([item[0] for item in batch_buffer], dim=0)  # (B,C,T,H,W)
    out_paths = [item[1] for item in batch_buffer]

    with torch.no_grad():
        features = model(tensors).cpu().numpy()  # (B, F)

    for feat, path in zip(features, out_paths):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, feat)
        print(f"✅ Guardado: {os.path.basename(path)}")



    


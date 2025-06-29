import os
import cv2
import torch
import numpy as np
from torchvision.transforms import Resize, CenterCrop, Normalize
from tqdm import tqdm

# Normalización estándar
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]

def extract_features_folder(
    input_folder: str,
    output_folder: str,
    model_name: str,
    sampling_rate: int,
    device: str = 'cuda',
    batch_size: int = 1,
    num_frames: int = 1,
):
    MODEL_PARAMS = {
        'x3d_xs': {'side_size': 182, 'crop_size': 182, 'num_frames': num_frames},
        'x3d_s':  {'side_size': 182, 'crop_size': 182, 'num_frames': num_frames},
        'x3d_m':  {'side_size': 256, 'crop_size': 256, 'num_frames': num_frames},
        'x3d_l':  {'side_size': 320, 'crop_size': 320, 'num_frames': num_frames},
    }

    os.makedirs(output_folder, exist_ok=True)
    assert model_name in MODEL_PARAMS, f"Modelo no soportado: {model_name}"
    params = MODEL_PARAMS[model_name]

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Usando dispositivo: {device}")

    # Modelo
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    model.eval().to(device)
    if hasattr(model, 'blocks'):
        del model.blocks[-1]

    # Transformaciones en GPU
    resize = Resize(params['side_size'])
    centercrop = CenterCrop((params['crop_size'], params['crop_size']))
    normalize = Normalize(MEAN, STD)

    files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    files.sort()

    batch_tensor_list = []
    batch_paths = []

    pbar = tqdm(files, desc='🔄 Extrayendo features')

    for fname in pbar:
        base, _ = os.path.splitext(fname)
        out_path = os.path.join(output_folder, base + '.npy')
        if os.path.exists(out_path):
            continue

        video_path = os.path.join(input_folder, fname)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(frame_rgb)
        cap.release()

        if len(frames) < params['num_frames']:
            continue

        sampled = frames[::sampling_rate]
        if len(sampled) < params['num_frames']:
            sampled += [sampled[-1]] * (params['num_frames'] - len(sampled))
        else:
            sampled = sampled[:params['num_frames']]

        # Transformar frames a GPU
        torch_clip = []
        for img in sampled:
            tensor = torch.from_numpy(img).permute(2, 0, 1).to(device)  # C,H,W
            tensor = normalize(centercrop(resize(tensor)))
            torch_clip.append(tensor)

        video_tensor = torch.stack(torch_clip, dim=1).unsqueeze(0)  # (1, C, T, H, W)
        batch_tensor_list.append(video_tensor)
        batch_paths.append(out_path)

        if len(batch_tensor_list) >= batch_size:
            _process_batch(batch_tensor_list, batch_paths, model)
            batch_tensor_list = []
            batch_paths = []

    if batch_tensor_list:
        _process_batch(batch_tensor_list, batch_paths, model)

def _process_batch(tensor_batch, out_paths, model):
    batch_tensor = torch.cat(tensor_batch, dim=0)  # (B, C, T, H, W)

    with torch.no_grad():
        features = model(batch_tensor)  # (B, F)

    for feat, path in zip(features, out_paths):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, feat.detach().cpu().numpy())

import subprocess
import torch
import os
from extractor import extract_features_folder
from validacion import evaluar_todos_los_modelos  # Asegúrate de tener esta función en ese módulo

ckpt_dir = './ckpt'
modelos_entrenados = [f.replace(".pkl", "") for f in os.listdir(ckpt_dir) if f.endswith(".pkl")]

for model in ['fast', 'base']:  # Empieza con 'fast'
    for size in ['xs', 's', 'm']:
        for frame in (4, 8, 16, 20, 24, 32):
            skip_fast_xs_4 = (model == 'fast' and size == 'xs' and frame == 4)
            if skip_fast_xs_4:
                print("⏭️ Saltando fast_xs_4 como pediste.")
                continue

            for ventana in range(12, 84, 8):
                stride = int(ventana / frame)
                output = f'features_x3d_{size}_{ventana}fps_{frame}_{stride}'
                model_name = f'STEAD_{model.upper()}_{size.upper()}_{frame}_{stride}'

                if model_name + "final" in modelos_entrenados:
                    print(f"✅ Ya existe {model_name}, se omite.")
                    continue

                print(f"\n🔧 Extrayendo características para: {output}")
                extract_features_folder(
                    input_folder='DATASET',
                    output_folder=output,
                    sampling_rate=stride,
                    num_frames=frame,
                    model_name=f'x3d_{size}',
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    batch_size=64
                )

                print(f"\n🚀 Ejecutando entrenamiento para: {model_name}")
                command = [
                    "python", "git_main.py",
                    "--comment", "vf_1",
                    "--batch_size", "64",
                    "--dataset_path", output,
                    "--model_name", model_name,
                    "--model_arch", model
                ]
                subprocess.run(command)

            # 🔎 Validar todos los modelos nuevos tras terminar con este `frame`
            print(f"\n📊 Evaluando modelos después de entrenar todos con frame={frame}")
            evaluar_todos_los_modelos()
import os
import pandas as pd

def save_results(experimento, itr, auc_train, auc_test, auc_test_l,auc_test_m, auc_test_s):
    # Crear carpeta si no existe
    path = os.path.join("results")
    os.makedirs(path, exist_ok=True)

    # Ruta del archivo Excel
    excel_file = os.path.join(path, f"{experimento}.xlsx")

    # Crear un nuevo DataFrame con los datos actuales
    new_data = pd.DataFrame([{
        "ITR": itr,
        "AUC_TRAIN": round(auc_train, 4),
        "AUC_TEST": round(auc_test, 4),
        "AUC_TEST_L": round(auc_test_l, 4),
        "AUC_TEST_M": round(auc_test_m, 4),
        "AUC_TEST_S": round(auc_test_s, 4)
    }])

    if os.path.exists(excel_file):
        # Si ya existe, cargar y agregar
        df = pd.read_excel(excel_file)
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        # Si no existe, este será el primer registro
        df = new_data

    # Guardar el DataFrame en Excel
    df.to_excel(excel_file, index=False)

    return 0


import os
import re

def save_results_validation(model_name, auc_test, auc_l, auc_m, auc_s, results_path="results/summary.xlsx"):
    # Extraer parámetros del nombre del modelo con regex
    match = re.match(r"STEAD_(\w+)_(\w+)_(\d+)_(\d+)final", model_name)
    if not match:
        print(f"❌ Nombre de modelo inválido: {model_name}")
        return

    arch, size, num_frames, stride = match.groups()
    num_frames = int(num_frames)
    stride = int(stride)

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Cargar si existe
    if os.path.exists(results_path):
        df = pd.read_excel(results_path)
    else:
        df = pd.DataFrame(columns=[
            "model_name", "arch", "size", "num_frames", "stride",
            "best_auc", "auc_l", "auc_m", "auc_s"
        ])

    # Revisar si ya existe el modelo
    if model_name in df["model_name"].values:
        idx = df.index[df["model_name"] == model_name][0]
        df.loc[idx] = {
            "model_name": model_name,
            "arch": arch,
            "size": size,
            "num_frames": num_frames,
            "stride": stride,
            "best_auc": round(auc_test, 4),
            "auc_l": round(auc_l, 4),
            "auc_m": round(auc_m, 4),
            "auc_s": round(auc_s, 4)
        }
    else:
        # Agregar nuevo modelo
        df = pd.concat([
            df,
            pd.DataFrame([{
                "model_name": model_name,
                "arch": arch,
                "size": size,
                "num_frames": num_frames,
                "stride": stride,
                "best_auc": round(auc_test, 4),
                "auc_l": round(auc_l, 4),
                "auc_m": round(auc_m, 4),
                "auc_s": round(auc_s, 4)
            }])
        ], ignore_index=True)

    # Guardar
    df.to_excel(results_path, index=False)
    print(f"✅ Resultados actualizados en: {results_path}")



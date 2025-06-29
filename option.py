import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='STEAD')

    # Argumentos de entrenamiento y otros (ya existentes)

    parser.add_argument('--dataset_path', default='exp_1', help='name of the model')


    parser.add_argument('--comment', default='tiny', help='comment for the ckpt name of the training')

    parser.add_argument('--dropout_rate', type=float, default=0.4, help='dropout rate')
    parser.add_argument('--attn_dropout_rate', type=float, default=0.1, help='attention dropout rate')
    parser.add_argument('--lr', type=str, default=2e-4, help='learning rates for steps default:2e-4')
    parser.add_argument('--batch_size', type=int, default=32, help='number of instances in a batch of data (default: 16)')

    parser.add_argument('--model_name', default='model', help='name to save model')
    parser.add_argument('--model_arch', default='fast', help='base or fast')
    parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model (for training)')
    parser.add_argument('--max_epoch', type=int, default=100, help='maximum iteration to train (default: 10)')
    parser.add_argument('--warmup', type=int, default=1, help='number of warmup epochs')

    # --- Argumentos para inferencia / procesamiento de video ---
    parser.add_argument('--model_path', type=str, required=False, help='Nombre del modelo personalizado (ej: 888tiny)')
    parser.add_argument('--video_path', type=str, required=False, help='Path al video que se desea procesar')
    parser.add_argument('--x3d', type=str, choices=['xs', 's', 'm', 'l'], required=False, help='Versin del modelo X3D')
    parser.add_argument('--output', type=str, required=False, help='Path para el video de salida con grfico')

# Argumentos adicionales para inferencia
    parser.add_argument('--save_name', type=str, default='inferencia', help='Nombre base del archivo de video de salida (sin .mp4)')
    parser.add_argument('--visualize', action='store_true', help='Mostrar visualizacin en tiempo real durante inferencia')
    parser.add_argument('--model_name_suffix', type=str, required=False, help='Nombre del modelo personalizado (ej: 888tiny)')
    args = parser.parse_args()
    return args

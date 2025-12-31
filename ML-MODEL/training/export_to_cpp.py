import numpy as np
import json

def export_weights_to_cpp():
    print("=" * 60)
    print("Exporting Model Weights to C++ Header File")
    print("=" * 60)

    encoder_weights = np.load('../models/encoder_weights.npy')
    encoder_bias = np.load('../models/encoder_bias.npy')
    decoder_weights = np.load('../models/decoder_weights.npy')
    decoder_bias = np.load('../models/decoder_bias.npy')

    with open('../models/model_config.json', 'r') as f:
        config = json.load(f)

    cpp_code = f'''#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#define INPUT_DIM {config["input_dim"]}
#define HIDDEN_DIM {config["hidden_dim"]}
#define RECONSTRUCTION_THRESHOLD {config["threshold"]:.4f}f

const float encoder_weights[INPUT_DIM][HIDDEN_DIM] = {{
'''

    for i in range(encoder_weights.shape[0]):
        cpp_code += '  {'
        for j in range(encoder_weights.shape[1]):
            cpp_code += f'{encoder_weights[i, j]:.6f}f'
            if j < encoder_weights.shape[1] - 1:
                cpp_code += ', '
        cpp_code += '}'
        if i < encoder_weights.shape[0] - 1:
            cpp_code += ','
        cpp_code += '\n'

    cpp_code += '};\n\n'

    cpp_code += f'const float encoder_bias[HIDDEN_DIM] = {{\n  '
    cpp_code += ', '.join([f'{b:.6f}f' for b in encoder_bias])
    cpp_code += '\n};\n\n'

    cpp_code += 'const float decoder_weights[HIDDEN_DIM][INPUT_DIM] = {\n'
    for i in range(decoder_weights.shape[0]):
        cpp_code += '  {'
        for j in range(decoder_weights.shape[1]):
            cpp_code += f'{decoder_weights[i, j]:.6f}f'
            if j < decoder_weights.shape[1] - 1:
                cpp_code += ', '
        cpp_code += '}'
        if i < decoder_weights.shape[0] - 1:
            cpp_code += ','
        cpp_code += '\n'

    cpp_code += '};\n\n'

    cpp_code += f'const float decoder_bias[INPUT_DIM] = {{\n  '
    cpp_code += ', '.join([f'{b:.6f}f' for b in decoder_bias])
    cpp_code += '\n};\n\n'

    cpp_code += f'const float scaler_mean[INPUT_DIM] = {{\n  '
    cpp_code += ', '.join([f'{m:.6f}f' for m in config["scaler_mean"]])
    cpp_code += '\n};\n\n'

    cpp_code += f'const float scaler_std[INPUT_DIM] = {{\n  '
    cpp_code += ', '.join([f'{s:.6f}f' for s in config["scaler_std"]])
    cpp_code += '\n};\n\n'

    cpp_code += '''float runInference(float features[INPUT_DIM]) {
  float scaled[INPUT_DIM];
  for(int i = 0; i < INPUT_DIM; i++) {
    scaled[i] = (features[i] - scaler_mean[i]) / scaler_std[i];
  }

  float hidden[HIDDEN_DIM] = {0};
  for(int j = 0; j < HIDDEN_DIM; j++) {
    for(int i = 0; i < INPUT_DIM; i++) {
      hidden[j] += scaled[i] * encoder_weights[i][j];
    }
    hidden[j] += encoder_bias[j];
    if(hidden[j] < 0) hidden[j] = 0;
  }

  float reconstructed[INPUT_DIM] = {0};
  for(int i = 0; i < INPUT_DIM; i++) {
    for(int j = 0; j < HIDDEN_DIM; j++) {
      reconstructed[i] += hidden[j] * decoder_weights[j][i];
    }
    reconstructed[i] += decoder_bias[i];
  }

  float error = 0;
  for(int i = 0; i < INPUT_DIM; i++) {
    float diff = scaled[i] - reconstructed[i];
    error += diff * diff;
  }
  return error / INPUT_DIM;
}

bool isAnomalous(float features[INPUT_DIM]) {
  float reconError = runInference(features);
  return (reconError > RECONSTRUCTION_THRESHOLD);
}

#endif
'''

    output_path = '../exports/model_weights.h'
    with open(output_path, 'w') as f:
        f.write(cpp_code)

    print(f"\nC++ header file exported to: {output_path}")
    print(f"\nModel Statistics:")
    print(f"  Input Dimensions: {config['input_dim']}")
    print(f"  Hidden Dimensions: {config['hidden_dim']}")
    print(f"  Reconstruction Threshold: {config['threshold']:.4f}")
    print(f"  Total Parameters: {encoder_weights.size + encoder_bias.size + decoder_weights.size + decoder_bias.size}")

    total_bytes = (encoder_weights.size + encoder_bias.size +
                   decoder_weights.size + decoder_bias.size +
                   len(config['scaler_mean']) + len(config['scaler_std'])) * 4
    print(f"  Model Size: ~{total_bytes} bytes ({total_bytes/1024:.2f} KB)")

    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print("\nTo use in ESP32:")
    print("  1. Copy model_weights.h to your ESP32 project")
    print("  2. #include \"model_weights.h\"")
    print("  3. Call runInference(features) or isAnomalous(features)")
    print("\n✅ Ready for embedded deployment\n")

if __name__ == "__main__":
    export_weights_to_cpp()

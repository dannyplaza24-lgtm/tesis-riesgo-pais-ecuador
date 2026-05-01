import json
import sys

notebooks = [
    '01_Data_Acquisition.ipynb',
    '02_Modeling_Baseline.ipynb', 
    '03_Feature_Engineering_Selection.ipynb',
    '04_Modelling_ML_SHAP.ipynb',
    '05_Ablation_Experiment.ipynb',
    '06_Granger_Causality.ipynb',
    '07_Regimenes_Estructurales.ipynb',
    '08_Autoencoder_NLP.ipynb'
]

for nb_path in notebooks:
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        print(f"\n{'='*80}")
        print(f"NOTEBOOK: {nb_path}")
        print(f"{'='*80}")
        
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                # Look for key patterns
                if any(x in source for x in ['target_future', 'StandardScaler', 'fit(', 'select_', 'train_test_split', '\.fit.*train', 'forward_fill', 'ffill']):
                    print(f"\nCELL {i}:")
                    lines = source.split('\n')
                    for j, line in enumerate(lines):
                        if j < 50:
                            print(line)
                        if len(lines) > 50:
                            print("...")
                            break
    except Exception as e:
        print(f"Error reading {nb_path}: {e}")


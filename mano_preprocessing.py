import argparse
import numpy as np
import pickle

# This code is used to
# 1. convert the chumpy objects of the official MANO model into numpy arrays
# 2. format the fields expected by models/MANO.py

src_right = './assets/MANO_RIGHT.pkl'
src_left  = './assets/MANO_LEFT.pkl'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess MANO model')
    parser.add_argument('--src_right', type=str, default=src_right)
    parser.add_argument('--src_left',  type=str, default=src_left)
    parser.add_argument('--output_dir', type=str, default='./assets')
    args = parser.parse_args()

    for side, src_path in [('RIGHT', args.src_right), ('LEFT', args.src_left)]:
        with open(src_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        v_template = np.array(data['v_template'])     # (778, 3)
        shapedirs  = np.array(data['shapedirs'])       # (778, 3, 10)
        posedirs   = np.array(data['posedirs'])        # (778, 3, 135)
        weights    = np.array(data['weights'])         # (778, 16)
        faces      = np.array(data['f'])               # (1538, 3)

        # J_regressor: dense (16, 778)
        J_regressor = np.array(data['J_regressor'].todense())

        # Build 21-joint regressor by appending one-hot fingertip rows to the 16-joint one.
        # Fingertip vertex indices (thumb→pinky) derived from the existing processed model.
        fingertip_vertices = [745, 317, 444, 673, 556]
        fingertip_rows = np.zeros((5, 778), dtype=np.float64)
        for i, v in enumerate(fingertip_vertices):
            fingertip_rows[i, v] = 1.0
        J_regressor_21 = np.vstack([J_regressor, fingertip_rows])  # (21, 778)

        # kintree_table[0] is the parent index array (16,); fix root to point to itself
        parents_16 = data['kintree_table'][0].astype(np.int32)
        parents_16[0] = 0
        # kintree_table_J for 21 joints: fingertips are children of the last joints of each finger
        finger_tip_parents = [15, 3, 6, 12, 9]  # distal joints for each finger
        parents_21 = np.concatenate([parents_16, finger_tip_parents]).astype(np.int32)

        model = {
            'v_template':      v_template,
            'shapedirs':       shapedirs,
            'posedirs':        posedirs,
            'weights':         weights,
            'f':               faces,
            'J_regressor':     np.array(J_regressor),         # (16, 778) dense matrix
            'kintree_table':   parents_16,                     # (16,) parent indices
            'kintree_table_J': parents_21,                     # (21,) parent indices
        }

        if side == 'RIGHT':
            model['J_regressor_right'] = np.array(J_regressor_21)   # (21, 778) dense matrix
        else:
            model['J_regressor_left']  = np.array(J_regressor_21)   # (21, 778) dense matrix

        output_path = f'{args.output_dir}/processed_MANO_{side}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        print(f'Saved {output_path}')

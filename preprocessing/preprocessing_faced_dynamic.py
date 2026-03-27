import scipy
from scipy import signal
import os
import lmdb
import pickle
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--s_duration', type=int, default=10, help='segment duration for training (default: 10)')
    parser.add_argument('--label_npy', type=str, default='./data/emotion_matrix.npy',
                        help='path to emotion_matrix.npy (shape: 28, 15, 7)')
    params = parser.parse_args()
    print(params)
    labels = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8])
    emotion_matrix = np.load(params.label_npy).astype(np.float32) # [28, 15, 7]
    root_dir = '/mnt/dataset0/qingzhu/EEG_raw/FACED/Processed_data'
    files = [file for file in os.listdir(root_dir)]
    files = sorted(files)

    files_dict = {
        'train':files[:80],
        'val':files[80:100],
        'test':files[100:],
    }

    dataset = {
        'train': list(),
        'val': list(),
        'test': list(),
    }

    s_duration = params.s_duration
    db = lmdb.open(F'./data/datasets/BigDownstream/Faced/processed_dynamic_{s_duration}s', map_size=6612500172)
    for files_key in files_dict.keys():
        for file in files_dict[files_key]:
            f = open(os.path.join(root_dir, file), 'rb')
            array = pickle.load(f)
            eeg = signal.resample(array, 6000, axis=2)
            eeg_ = eeg.reshape(28, 32, 30, 200)
            for i, (samples, label) in enumerate(zip(eeg_, labels)):
                trial_emotion = emotion_matrix[i] # [15, 7]
                # interpolate to 30s
                trial_emotion_interp = signal.resample(trial_emotion, 30, axis=0) # [30, 7]
                for j in range(30 // s_duration):
                    sample = samples[:, s_duration*j:s_duration*(j+1), :]
                    sample_key = f'{file}-{i}-{j}'
                    print(sample_key)
                    # label is the average emotion in this segment
                    segment_emotion = trial_emotion_interp[s_duration*j:s_duration*(j+1), :].mean(axis=0)
                    data_dict = {
                        'sample': sample, 'label': segment_emotion
                    }
                    txn = db.begin(write=True)
                    txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                    txn.commit()
                    dataset[files_key].append(sample_key)


    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
    txn.commit()
    db.close()

if __name__ == '__main__':
    main()
from emg2pose.data import Emg2PoseSessionData
import numpy as np

def emg_features(x_win, zc_thresh=1e-3, ssc_thresh=1e-3):
    # MAV
    mav = np.mean(np.abs(x_win), axis=0)

    # RMS
    rms = np.sqrt(np.mean(x_win**2, axis=0))

    # WL
    wl = np.sum(np.abs(np.diff(x_win, axis=0)), axis=0)

    # ZC
    zc = np.sum(
        (np.diff(np.sign(x_win), axis=0) != 0) &
        (np.abs(np.diff(x_win, axis=0)) > zc_thresh),
        axis=0
    )

    # SSC
    diff1 = np.diff(x_win, axis=0)
    diff2 = np.diff(diff1, axis=0)
    ssc = np.sum(
        (diff1[:-1] * diff1[1:] < 0) &
        (np.abs(diff2) > ssc_thresh),
        axis=0
    )

    # concatenate all features
    return np.concatenate([mav, rms, wl, zc, ssc])

def features(data : Emg2PoseSessionData):

    window = 500      # 250 ms @ 2000 Hz
    stride = 67       # ~30 Hz

    X = data['emg']           
    y = data['joint_angles']

    X_feats = []
    y_out = []

    for start in range(0, len(X) - window + 1, stride):
        end = start + window

        x_win = X[start:end]
        y_target = y[end - 1]   # align gt to end of window

        # placeholder feature (you replace this)
        feats = emg_features(x_win)

        X_feats.append(feats)
        y_out.append(y_target)

    X_feats = np.array(X_feats)
    y_out = np.array(y_out)

    return X_feats, y_out
    
if __name__ == "__main__":
    from pathlib import Path
    import glob
    import os

    DATA_DOWNLOAD_DIR = Path("/Volumes") / "Crucial X9" # replace with your dataset path

    # Choose a user
    users = sorted([
        p for p in Path(DATA_DOWNLOAD_DIR, "emg2pose_dataset_mini").iterdir()
        if p.is_dir()
    ])
    user = users[0] 
    
    sessions = sorted(glob.glob(os.path.join(user, "*.hdf5")))

    # Select our session 
    session = sessions[0]
    data = Emg2PoseSessionData(hdf5_path=session)

    X_feats, y_out = features(data)

    print(X_feats.shape)
    print(y_out.shape)
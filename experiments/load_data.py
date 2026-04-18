import random
from emg2pose.data import Emg2PoseSessionData

# Training User and Session Selection Helpers
def _user_has_valid_session(user):
    for s in user.glob("*.hdf5"):
        try:
            _ = Emg2PoseSessionData(hdf5_path=s)
            return True
        except:
            continue
    return False

def pick_one_user(user_list):
    while True:
        rand_user = random.choice(user_list)
        if _user_has_valid_session(rand_user):
            return {rand_user: []}

def random_subset(user_list, k):
    selected = {}

    while len(selected) < k:
        rand_user = random.choice(user_list)

        if rand_user in selected:
            continue

        if _user_has_valid_session(rand_user):
            selected[rand_user] = []

    return selected

def pick_sessions(data_regime, user_train_dict):
    for user in user_train_dict:
        all_sessions = sorted(user.glob("*.hdf5"))

        valid_sessions = []
        for s in all_sessions:
            try:
                _ = Emg2PoseSessionData(hdf5_path=s)
                valid_sessions.append(s)
            except:
                continue

        if not valid_sessions:
            raise RuntimeError(f"No valid sessions for {user}")

        if data_regime == "single_session":
            user_train_dict[user] = [random.choice(valid_sessions)]
        else:
            user_train_dict[user] = valid_sessions

    return user_train_dict

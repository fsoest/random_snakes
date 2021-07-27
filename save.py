import pathlib
import pickle


OBJ_DIR = pathlib.Path(__file__).parent / 'obj'


def save_obj(obj, name):
    with open(OBJ_DIR / f'{name.pkl}', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(OBJ_DIR / f'{name.pkl}', 'rb') as f:
        return pickle.load(f)

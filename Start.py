import os
from models import SIRM_deterministic
from models import SIRM_deterministic_T2


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

if __name__=="__main__":
    file_path = "models_pickle"
    ensure_dir(file_path)

    objekat = SIRM_deterministic()
    objekat.fit(iter_max=50,pop_size=300)
    objekat.predict()
    objekat.evaluate()
    objekat.plot()

    objekat.save('{}/model_{}.pkl'.format(file_path, objekat.name))
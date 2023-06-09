import pandas as pd
from biopandas.pdb import PandasPdb

x = pd.read_pickle('/apdcephfs/share_1364275/kaithgao/flexdock_0425/complexes_0.dill')
ml = list(x['data'])
f = open("/apdcephfs/share_1364275/kaithgao/flexdock_0425/list.txt", "w")
for item in ml:
    f.write(str(item) + ",")
f.close()

a = 1
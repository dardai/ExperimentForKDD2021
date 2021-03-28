import numpy as np
import pandas as pd

txt = np.loadtxt('u.data')
txtDF = pd.DataFrame(txt)
txtDF.to_csv('file.csv', index=False)


if __name__ == '__main__':
    print ("hello world")
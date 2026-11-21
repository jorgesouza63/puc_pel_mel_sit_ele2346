"""Função oficial de concatenação das imagens acústicas do WAID.

ORIGEM : https://github.com/petrobras/waid (README.md, seção "Acoustic image logs dataset")
COMMIT : f8926d060cac24cdee37563546db85daf06a2859 (18/06/2026)
LICENÇA: Apache License 2.0 (código do projeto WAID, PETROBRAS)
ALTERAÇÕES: nenhuma. O código abaixo é cópia literal da função original.

Atenção: a função lê os subarquivos na ordem de os.listdir(), que depende do
sistema de arquivos. Por isso o projeto usa o invólucro load_well_image(), em
src/data/loader.py, que ordena e verifica a profundidade.
"""
import pandas as pd
import numpy as np
import os

def concat_IMG_data(well_id, data_path):
    # Due to file size limitations, the original AMP '.csv' file
    # has been split into several sub-files.
    # The concat_IMG_data() function aims to concatenate
    # them back into a single data object.
    #
    # concat_IMG_data() returns 'image_df', a Pandas dataframe
    # indexed by DEPTH information and whose columns are
    # the azimuthal coordinates of the AMP image log.
    
    # Name of the initial '00' file
    initial_file = well_id + "_AMP00.csv"

    # Read the the initial file to capture header information
    initial_file_path = os.path.join(data_path, initial_file)
    image_df = pd.read_csv(initial_file_path,sep = ';',
                           index_col=0,
                           na_values = -9999,na_filter = True,
                           decimal = ',',
                           skip_blank_lines = True).dropna()

    # Read and add data from the remaining files sequentially
    for file in os.listdir(data_path):
        if file.startswith(well_id) and file != initial_file:
            file_path = os.path.join(data_path, file)
            df_temp = pd.read_csv(file_path,sep = ';',
                                  header=None,index_col = 0,
                                  na_values = -9999, na_filter = True,
                                  decimal = ',', skip_blank_lines = True,
                                  dtype=np.float32
                                 ).dropna()
            
            # Adjust tem df's header to match image header
            df_temp.columns=image_df.columns
            
            # Concat dfs
            image_df = pd.concat([image_df, df_temp])
    return image_df

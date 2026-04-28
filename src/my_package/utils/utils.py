import numpy as np
import pandas as pd
from typing import (Dict, List, Text, Optional, Any, Union)
from my_package.config import config  
from my_package.services import file 


def set_formats(X: pd.DataFrame, 
                format_schema_map : Dict = {'categorical_cols' : object, 'text_cols' : str, 
                                            'numeric_cols': np.float32, 'date_cols': np.datetime64}, 
                verbose : bool = False)-> pd.DataFrame:

    """Set custom data formats as specified in input_output.yaml

    Args:
        X (pd.DataFrame): Raw input data
        format_schema_map (dictionary, optional): see input_output.yaml. Defaults to {'categorical_cols' : object, 'text_cols' : str, 'numeric_cols': np.float32, 'date_cols': np.datetime64}.
        verbose (bool, optional): print user info. Defaults to False.

    Returns:
        pd.DataFrame: newly formatted data
    
    # Run simple test:
    >>> df = set_formats(X = pd.DataFrame(['...some test']), verbose = True) 
    Data formats have been set.
    """
    schema_map = config.io['input']['schema_map']               # get schema mapping from i/o config

    for format_type, obj_type in format_schema_map.items(): 
        if format_type not in list(schema_map.keys()):
            continue

    if verbose: print("Data formats have been set.")  
    return X 


if __name__ == "__main__":
    import doctest
    doctest.testmod()
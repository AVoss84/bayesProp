"""
Services for reading and writing from and to AWS S3 of various file formats
"""
from typing import (List, Optional, Callable)
import os, toml, boto3
from io import (BytesIO, StringIO)
import pandas as pd
from my_package.config import global_config as glob
from my_package.services import file
#from imp import reload


def list_files(bucket: Callable, path : str = "")-> List:
    """
    List all files from s3 bucket directory.
    
    bucket: bucket object instance - boto3.resources.factory.s3.Bucket
    path: s3 bucket directory name
    """
    try:
        return [obj.key for obj in bucket.objects.filter(Prefix=path)]
    except Exception as ex:
        print(ex)
        return list()

class CSVService:
    
    def __init__(self, path : Optional[str] = "", delimiter : str = None, encoding : str = "UTF-8", schema_map : Optional[dict] = None, 
                 root_path : str = glob.UC_DATA_DIR, verbose : bool = True):
        """Read/write service instance for CSV files
        Args:
            path (str, optional): Filename. Defaults to "".
            delimiter (str, optional): see pd.read_csv. Defaults to "\t".
            encoding (str, optional): see pd.read_csv. Defaults to "UTF-8".
            schema_map (Optional[dict], optional): mapping scheme for renaming of columns, see pandas rename. Defaults to None.
            root_path (str, optional): root path where file is located. Defaults to glob.UC_DATA_DIR.
            verbose (bool, optional): should user information be displayed? Defaults to False.
        """
        self.path = os.path.join(root_path, path)
        self.delimiter = delimiter
        self.verbose = verbose
        self.encoding = encoding
        self.schema_map = schema_map
        self.aws_cred = file.TOMLservice(root_path = glob.UC_AWS_DIR, path = "aws_config.toml").doRead()
    
    def doRead(self, **kwargs)-> pd.DataFrame:
        """Read data from CSV
        Returns:
            pd.DataFrame: data converted to dataframe
        """
        try: 
            s3 = boto3.resource('s3')
            bucket = s3.Bucket(self.aws_cred['Credentials']['bucket_name'])
            self.s3obj = bucket.Object(self.path).get()
            bytes_data = self.s3obj['Body'].read()
            df = pd.read_csv(BytesIO(bytes_data), encoding=self.encoding, delimiter=self.delimiter, **kwargs)
            #df = pd.read_csv(self.s3obj['Body'], index_col=0, encoding=self.encoding, delimiter=self.delimiter, **kwargs)
            if self.verbose: 
                print(f"CSV Service Read from File: {str(self.path)}")
            if self.schema_map: df.rename(columns=self.schema_map, inplace=True)
            return df
        except Exception as ex:
            print(ex)

    def doWrite(self, X : pd.DataFrame, **kwargs):
        """Write X to CSV file
        Args:
            X (pd.DataFrame): input data
        """
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.aws_cred['Credentials']['bucket_name'])
        csv_buffer = StringIO()
        X.to_csv(csv_buffer, index=False)
        try:
            self.response = s3.Object(bucket.name, self.path).put(Body=csv_buffer.getvalue())
            if self.verbose: 
                print(f"CSV Service Output to File: {str(self.path)}")
        except Exception as ex:
            print(ex)

            
class ParquetService:
    
    def __init__(self, path : Optional[str] = "", schema_map : Optional[dict] = None, 
                 root_path : str = glob.UC_DATA_DIR, verbose : bool = True):
        """Read/write service instance for parquet files
        Args:
            path (str, optional): Filename. Defaults to "".
            schema_map (Optional[dict], optional): mapping scheme for renaming of columns, see pandas rename. Defaults to None.
            root_path (str, optional): root path where file is located. Defaults to glob.UC_DATA_DIR.
            verbose (bool, optional): should user information be displayed? Defaults to False.
        """
        self.path = os.path.join(root_path, path)
        self.verbose = verbose
        self.schema_map = schema_map
        self.aws_cred = file.TOMLservice(root_path = glob.UC_AWS_DIR, path = "aws_config.toml").doRead()
    
    def doRead(self, **kwargs)-> pd.DataFrame:
        """Read data from parquet
        
        Returns:
            pd.DataFrame: data converted to dataframe
        """
        try: 
            s3 = boto3.resource('s3')
            bucket = s3.Bucket(self.aws_cred['Credentials']['bucket_name'])
            self.s3obj = bucket.Object(self.path).get()
            bytes_data = self.s3obj['Body'].read()
            df = pd.read_parquet(BytesIO(bytes_data), **kwargs)
            if self.verbose: 
                print(f"Parquet service read from s3: {str(self.path)}")
            if self.schema_map: 
                df.rename(columns=self.schema_map, inplace=True)
            return df
        except Exception as ex:
            print(ex)

    def doWrite(self, X : pd.DataFrame, **kwargs):
        """Write X to parquet file
        Args:
            X (pd.DataFrame): input data
        """
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.aws_cred['Credentials']['bucket_name'])
        buffer = BytesIO()
        X.to_parquet(buffer, index=False)
        try:
            self.response = s3.Object(bucket.name, self.path).put(Body=buffer.getvalue())
            if self.verbose: 
                print(f"Parquet service written to s3: {str(self.path)}")
        except Exception as ex:
            print(ex)


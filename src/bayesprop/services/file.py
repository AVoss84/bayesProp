"""Services for reading and writing from and to various file formats."""

import json
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

from bayesprop.config import global_config as glob
from bayesprop.services.blueprint_file import BaseService


class CSVService(BaseService):
    def __init__(
        self,
        path: str = "",
        root_path: str = glob.DATA_PKG_DIR,
        delimiter: str = "\t",
        encoding: str = "UTF-8",
        schema_map: Optional[dict] = None,
        verbose: bool = False,
    ):
        r"""Generic read/write service for CSV files.

        Args:
            path (str, optional): Filename. Defaults to "".
            delimiter (str, optional): see pd.read_csv. Defaults to "\t".
            encoding (str, optional): see pd.read_csv. Defaults to "UTF-8".
            schema_map (Optional[dict], optional): mapping scheme for renaming of columns, see pandas rename. Defaults to None.
            root_path (str, optional): root path where file is located. Defaults to glob.UC_DATA_DIR.
            verbose (bool, optional): should user information be displayed? Defaults to False.
        """
        super().__init__(path, root_path)
        self.path = os.path.join(root_path, path)
        # Ensure the directory exists
        if not os.path.exists(root_path):
            os.makedirs(root_path, exist_ok=True)
        self.delimiter = delimiter
        self.verbose = verbose
        self.encoding = encoding
        self.schema_map = schema_map

    def doRead(self, **kwargs: Any) -> pd.DataFrame:
        """Read data from CSV
        Args:
            **kwargs (Dict[str, Any]): Additional arguments for pandas read_csv.

        Returns:
            pd.DataFrame: data converted to dataframe
        """
        try:
            df = pd.read_csv(
                filepath_or_buffer=self.path,
                encoding=self.encoding,
                delimiter=self.delimiter,
                **kwargs,
            )
            if self.verbose:
                print(f"CSV Service read from file: {self.path}")
            if self.schema_map:
                df.rename(columns=self.schema_map, inplace=True)
            return df
        except Exception as e:
            print(f"Error reading CSV file {self.path}: {e}")
            return pd.DataFrame()

    def doWrite(self, X: pd.DataFrame, **kwargs: Any) -> None:
        """Write X to CSV file
        Args:
            X (pd.DataFrame): input data
        """
        try:
            X.to_csv(
                path_or_buf=self.path,
                encoding=self.encoding,
                sep=self.delimiter,
                index=False,
                **kwargs,
            )
            if self.verbose:
                print(f"CSV Service output to file: {self.path}")
        except Exception as e:
            print(f"Error writing CSV file {self.path}: {e}")


class YAMLService(BaseService):
    def __init__(
        self,
        path: str = "",
        root_path: str = glob.CODE_DIR,
        verbose: bool = False,
    ):
        """
        Generic read/write service for YAML files.

        Args:
            path (Optional[str]): Filename. Defaults to "".
            root_path (str): Root path where file is located. Defaults to glob.CODE_DIR.
            verbose (bool): Should user information be displayed? Defaults to False.
        """
        super().__init__(path, root_path)
        self.path = os.path.join(root_path, path)
        self.verbose = verbose

    def doRead(self, **kwargs: Any) -> Union[Dict, List]:
        """
        Read data from YAML file.

        Returns:
            Union[Dict, List]: Read-in YAML file.
        """
        with open(self.path, "r") as stream:
            try:
                my_yaml_load = yaml.safe_load(stream)
                if self.verbose:
                    print(f"Read: {self.path}")
            except yaml.YAMLError as exc:
                print(f"Error reading YAML file {self.path}: {exc}")
                raise
        return my_yaml_load

    def doWrite(self, X: Union[Dict, List], **kwargs: Any) -> None:
        """
        Write dictionary to YAML file.

        Args:
            X (Union[Dict, List]): Input data.
            **kwargs: Additional keyword arguments passed to ``yaml.dump``.
        """
        with open(self.path, "w") as outfile:
            try:
                yaml.dump(X, outfile, default_flow_style=False)
                if self.verbose:
                    print(f"Write to: {self.path}")
            except yaml.YAMLError as exc:
                print(f"Error writing YAML file {self.path}: {exc}")
                raise


class JSONService(BaseService):
    def __init__(self, path: str = "", root_path: str = glob.DATA_PKG_DIR, verbose: bool = True):
        """
        Generic read/write service for JSON files.

        Args:
            path (Optional[str]): Filename. Defaults to "".
            root_path (str): Root path where file is located. Defaults to "".
            verbose (bool): Should user information be displayed? Defaults to True.
        """
        super().__init__(path, root_path)
        self.path = os.path.join(root_path, path)
        self.verbose = verbose

    def doRead(self, **kwargs: Any) -> Dict:
        """
        Read data from JSON file.

        Returns:
            Dict: Output imported data.
        """
        if os.stat(self.path).st_size == 0:
            return dict()
        try:
            with open(self.path, "r") as stream:
                my_json_load = json.load(stream, **kwargs)
            if self.verbose:
                print(f"Read: {self.path}")
            return my_json_load
        except Exception as exc:
            print(f"Error reading JSON file {self.path}: {exc}")
            return {}

    def doWrite(self, X: Dict, **kwargs: Any) -> None:
        """
        Write dictionary to JSON file.

        Args:
            X (Dict): Input data.
            **kwargs: Additional keyword arguments passed to ``json.dump``.
        """
        with open(self.path, "w", encoding="utf-8") as outfile:
            try:
                json.dump(X, outfile, ensure_ascii=False, indent=4, **kwargs)
                if self.verbose:
                    print(f"JSON Service Output to File: {self.path}")
            except Exception as exc:
                print(f"Error writing JSON file {self.path}: {exc}")
                raise


class XLSXService:
    def __init__(
        self,
        path: Optional[str] = "",
        sheetname: str = "Sheet1",
        root_path: str = glob.DATA_PKG_DIR,
        schema_map: Optional[Dict[str, str]] = None,
        verbose: bool = False,
    ):
        """
        Generic read/write service for XLSX files.

        Args:
            path (Optional[str]): Filename. Defaults to "".
            sheetname (str): Sheet name for Excel file. Defaults to "Sheet1".
            root_path (str): Root path where file is located. Defaults to glob.DATA_DIR.
            schema_map (Optional[Dict[str, str]]): Mapping scheme for renaming columns. Defaults to None.
            verbose (bool): Should user information be displayed? Defaults to False.
        """
        self.path = os.path.join(root_path, path or "")
        self.writer = pd.ExcelWriter(self.path)
        self.sheetname = sheetname
        self.verbose = verbose
        self.schema_map = schema_map

    def doRead(self, **kwargs: Any) -> pd.DataFrame:
        """
        Read data from XLSX file.

        Returns:
            pd.DataFrame: Data converted to DataFrame.
        """
        df = pd.read_excel(self.path, self.sheetname, **kwargs)
        if self.verbose:
            print(f"XLSX Service Read from File: {str(self.path)}")
        if self.schema_map:
            df.rename(columns=self.schema_map, inplace=True)
        return df

    def doWrite(self, X: pd.DataFrame, **kwargs: Any) -> None:
        """
        Write DataFrame to XLSX file.

        Args:
            X (pd.DataFrame): Input data.
            **kwargs: Additional keyword arguments passed to ``DataFrame.to_excel``.
        """
        try:
            X.to_excel(self.writer, sheet_name=self.sheetname, index=False, **kwargs)
            self.writer.close()
            if self.verbose:
                print(f"XLSX Service Output to File: {self.path}")
        except Exception as e:
            print(f"Error writing XLSX file {self.path}: {e}")
            raise

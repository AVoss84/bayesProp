"""
Services for reading and writing from and to various file formats
"""

from typing import Dict, List, Optional, Union, Any
import os, yaml, json, toml, pickle
import pandas as pd
from my_package.config import global_config as glob


class CSVService:
    def __init__(
        self,
        path: Optional[str] = "",
        delimiter: str = "\t",
        encoding: str = "UTF-8",
        schema_map: Optional[Dict[str, str]] = None,
        root_path: str = glob.DATA_DIR,
        verbose: bool = False,
    ):
        """
        Generic read/write service for CSV files.

        Args:
            path (Optional[str]): Filename. Defaults to "".
            delimiter (str): Delimiter for CSV file. Defaults to "\t".
            encoding (str): Encoding for CSV file. Defaults to "UTF-8".
            schema_map (Optional[Dict[str, str]]): Mapping scheme for renaming columns. Defaults to None.
            root_path (str): Root path where file is located. Defaults to glob.DATA_DIR.
            verbose (bool): Should user information be displayed? Defaults to False.
        """
        self.path = os.path.join(root_path, path or "")
        self.delimiter = delimiter
        self.verbose = verbose
        self.encoding = encoding
        self.schema_map = schema_map

    def doRead(self, **kwargs: Any) -> pd.DataFrame:
        """
        Read data from CSV.

        Returns:
            pd.DataFrame: Data converted to DataFrame.
        """
        df = pd.read_csv(
            filepath_or_buffer=self.path,
            encoding=self.encoding,
            delimiter=self.delimiter,
            **kwargs,
        )
        if self.verbose:
            print(f"CSV Service Read from File: {str(self.path)}")
        if self.schema_map:
            df.rename(columns=self.schema_map, inplace=True)
        return df

    def doWrite(self, X: pd.DataFrame, **kwargs: Any) -> None:
        """
        Write DataFrame to CSV file.

        Args:
            X (pd.DataFrame): Input data.
        """
        X.to_csv(
            path_or_buf=self.path, encoding=self.encoding, sep=self.delimiter, **kwargs
        )
        if self.verbose:
            print(f"CSV Service Output to File: {str(self.path)}")


class XLSXService:
    def __init__(
        self,
        path: Optional[str] = "",
        sheetname: str = "Sheet1",
        root_path: str = glob.DATA_DIR,
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
        """
        X.to_excel(self.writer, self.sheetname, **kwargs)
        self.writer.save()
        if self.verbose:
            print(f"XLSX Service Output to File: {str(self.path)}")


class PickleService:
    def __init__(
        self,
        path: Optional[str] = "",
        root_path: str = glob.DATA_DIR,
        schema_map: Optional[Dict[str, str]] = None,
        is_df: bool = True,
        verbose: bool = True,
    ):
        """
        Generic read/write service for Pickle files.

        Args:
            path (Optional[str]): Filename. Defaults to "".
            is_df (bool): Is the data a pandas DataFrame? Defaults to True.
            root_path (str): Root path where file is located. Defaults to glob.DATA_DIR.
            schema_map (Optional[Dict[str, str]]): Mapping scheme for renaming columns. Defaults to None.
            verbose (bool): Should user information be displayed? Defaults to True.
        """
        self.path = os.path.join(root_path, path or "")
        self.schema_map = schema_map
        self.verbose = verbose
        self.is_df = is_df

    def doRead(self, **kwargs: Any) -> Union[pd.DataFrame, Any]:
        """
        Read data from Pickle file.

        Returns:
            Union[pd.DataFrame, Any]: Input data.
        """
        try:
            if self.is_df:
                data = pd.read_pickle(self.path, **kwargs)
                if self.schema_map:
                    data.rename(columns=self.schema_map, inplace=True)
            else:
                data = pickle.load(open(self.path, "rb"))
            if self.verbose:
                print(f"Pickle Service Read from file: {str(self.path)}")
            return data
        except Exception as e:
            print(e)
            return None

    def doWrite(self, X: Union[pd.DataFrame, Any], **kwargs: Any) -> None:
        """
        Write data to Pickle file.

        Args:
            X (Union[pd.DataFrame, Any]): Input data.
        """
        try:
            if self.is_df:
                X.to_pickle(path=self.path, compression=None)  # "gzip"
            else:
                pickle.dump(X, open(self.path, "wb"))
            if self.verbose:
                print(f"Pickle Service Output to file: {str(self.path)}")
        except Exception as e:
            print(e)


class YAMLService:
    def __init__(
        self,
        path: Optional[str] = "",
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
        self.path = os.path.join(root_path, path or "")
        self.verbose = verbose

    def doRead(self, **kwargs: Any) -> Union[Dict, List]:
        """
        Read data from YAML file.

        Returns:
            Union[Dict, List]: Read-in YAML file.
        """
        with open(self.path, "r") as stream:
            try:
                my_yaml_load = yaml.load(stream, Loader=yaml.FullLoader, **kwargs)
                if self.verbose:
                    print(f"Read: {self.path}")
            except yaml.YAMLError as exc:
                print(exc)
        return my_yaml_load

    def doWrite(self, X: Union[Dict, List], **kwargs: Any) -> None:
        """
        Write dictionary to YAML file.

        Args:
            X (Union[Dict, List]): Input data.
        """
        with open(self.path, "w") as outfile:
            try:
                yaml.dump(X, outfile, default_flow_style=False)
                if self.verbose:
                    print(f"Write to: {self.path}")
            except yaml.YAMLError as exc:
                print(exc)


class TXTService:
    def __init__(
        self,
        path: Optional[str] = "",
        root_path: str = glob.DATA_DIR,
        verbose: bool = True,
    ):
        """
        Generic read/write service for TXT files.

        Args:
            path (Optional[str]): Filename. Defaults to "".
            root_path (str): Root path where file is located. Defaults to glob.DATA_DIR.
            verbose (bool): Should user information be displayed? Defaults to True.
        """
        self.path = os.path.join(root_path, path or "")
        self.verbose = verbose

    def doRead(self, **kwargs: Any) -> List[str]:
        """
        Read data from TXT file.

        Returns:
            List[str]: Input data.
        """
        try:
            with open(self.path, **kwargs) as f:
                df = f.read().splitlines()
            if self.verbose:
                print(f"TXT Service read from file: {str(self.path)}")
        except Exception as e0:
            print(e0)
            df = []
        return df

    def doWrite(self, X: List[str], **kwargs: Any) -> None:
        """
        Write data to TXT file.

        Args:
            X (List[str]): Input data.
        """
        try:
            with open(self.path, "w", **kwargs) as f:
                f.write("\n".join(X))
            if self.verbose:
                print(f"TXT Service output to file: {str(self.path)}")
        except Exception as e0:
            print(e0)


class JSONService:
    def __init__(
        self, path: Optional[str] = "", root_path: str = "", verbose: bool = True
    ):
        """
        Generic read/write service for JSON files.

        Args:
            path (Optional[str]): Filename. Defaults to "".
            root_path (str): Root path where file is located. Defaults to "".
            verbose (bool): Should user information be displayed? Defaults to True.
        """
        self.path = os.path.join(root_path, path or "")
        self.verbose = verbose

    def doRead(self, **kwargs: Any) -> Dict:
        """
        Read data from JSON file.

        Returns:
            Dict: Output imported data.
        """
        if os.stat(self.path).st_size == 0:  # if JSON not empty
            return dict()
        try:
            with open(self.path, "r") as stream:
                my_json_load = json.load(stream, **kwargs)
            if self.verbose:
                print(f"Read: {self.path}")
            return my_json_load
        except Exception as exc:
            print(exc)
            return {}

    def doWrite(self, X: Dict, **kwargs: Any) -> None:
        """
        Write dictionary to JSON file.

        Args:
            X (Dict): Input data.
        """
        with open(self.path, "w", encoding="utf-8") as outfile:
            try:
                json.dump(X, outfile, ensure_ascii=False, indent=4, **kwargs)
                if self.verbose:
                    print(f"Write to: {self.path}")
            except Exception as exc:
                print(exc)


class TOMLService:
    def __init__(
        self,
        path: Optional[str] = "",
        root_path: str = glob.CODE_DIR,
        verbose: bool = False,
    ):
        """
        Generic read/write service for TOML files.

        Args:
            path (Optional[str]): Filename. Defaults to "".
            root_path (str): Root path where file is located. Defaults to glob.CODE_DIR.
            verbose (bool): Should user information be displayed? Defaults to False.
        """
        self.root_path = root_path
        self.path = path or ""
        self.verbose = verbose

    def doRead(self, **kwargs: Any) -> Dict:
        """
        Read data from TOML file.

        Returns:
            Dict: Imported TOML file.
        """
        with open(os.path.join(self.root_path, self.path), "r") as stream:
            try:
                toml_load = toml.load(stream, **kwargs)
                if self.verbose:
                    print(f"Read: {self.root_path + (self.path or '')}")
            except Exception as exc:
                print(exc)
                return {}
        return toml_load

    def doWrite(self, X: Dict, **kwargs: Any) -> None:
        """
        Write dictionary to TOML file.

        Args:
            X (Dict): Input dictionary.
        """
        with open(os.path.join(self.root_path, self.path), "w") as outfile:
            try:
                toml.dump(X, outfile)
                if self.verbose:
                    print(f"Write to: {self.root_path + (self.path or '')}")
            except Exception as exc:
                print(exc)
# excel_tools.py
import pandas as pd


class ExcelTools:
    """Utilities for handling Microsoft Excel files"""

    @staticmethod
    def dataframes_dictionary_from_excel_file(src_path: str):
        """Extract 'Excel tables' into a dictionary of DataFrames.

        Toy method, just to see how ExcelFile works.

        Assumes all desired data is in the form of a single block of data cells per worksheet.

        :param src_path: Full path to source Excel file
        :return: Dictionary of DataFrames
        """

        # read file
        wb = pd.ExcelFile(src_path)

        dfs = pd.read_excel(wb, sheet_name=None)

        return dfs


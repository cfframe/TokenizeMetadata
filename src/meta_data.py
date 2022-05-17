# meta_data.py
import pandas as pd
from src.meta_data_tools import MetaDataTools as MDT


class MetaData:
    """Meta Data File object"""
    def __init__(self, source_path: str):
        """A Meta Data File.

        Keyword arguments:
        :param source_path: Path to source data file
        """

        self.source_path = source_path
        """Path to original data."""

        self.raw_data = MDT.read_raw_data(source_path)
        """Original unprocessed data."""

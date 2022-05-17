# tokenize_meta_data.py
import argparse
import datetime

import numpy as np

from src.file_tools import FileTools
from src.meta_data_tools import MetaDataTools
import os
import pandas as pd
from pathlib import Path

TEST_FILEPATH = os.path.join(Path(__file__).parent, 'test', 'test_excel_files', 'test_xls.xlsx')

"""
Description: Extract fields, tokenized descriptors and labels from data dictionary. 

Example usage:
py tokenize_labelled_meta_data.py -s "C:/temp/TableMetaData/Source/FCRB_Data Model_v0.5 CFF 1g.xlsm" -td C:/temp/TableMetaData/Results 
py tokenize_labelled_meta_data.py -s "C:/temp/TableMetaData/Source/FCRB_Data Model_v0.5 CFF 1g.xlsm" -td C:/temp/TableMetaData/Results -ot bert
py tokenize_labelled_meta_data.py -s "C:/temp/TableMetaData/Source/FCRB_Data Model_v0.5 CFF 1g.xlsm" -td C:/temp/TableMetaData/Results -ot bert -tt
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Extract fields, descriptors and (optionally) tokenized text from '
                                                 'a labelled data dictionary.')
    parser.add_argument('-s', '--src_path', type=str, default=TEST_FILEPATH,
                        help='Source path for processing.')
    parser.add_argument('-td', '--target_dir', type=str, default=Path(__file__).parent,
                        help='Working directory for saving files etc')
    parser.add_argument('-ot', '--output_type', type=str, default='tsv',
                        help='Output type (tsv, bert')
    parser.add_argument('-tt', '--to_tokenize', action='store_true',
                        help='When used, indicates that descriptor text is to be tokenized.')
    parser.add_argument('-sn', '--stemmer_name', type=str, default='none',
                        help='Name of NLTK stemmer (default: none).')
    parser.add_argument('-sl', '--save_label', type=str, default='',
                        help='Optional label to add to save name for easier identification.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # Declare args - helps with auto-completion. Convert to object?
    src_path = args.src_path
    target_dir = args.target_dir
    output_type = args.output_type
    to_tokenize = args.to_tokenize
    save_label = args.save_label
    stemmer_name = args.stemmer_name

    if len(save_label) == 0:
        save_label = Path(src_path).stem[:3]

    prefix = f'{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}{save_label}'
    command_filename = f'{prefix} Command args.txt'
    FileTools.save_command_args_to_file(vars(args),
                                        save_path=os.path.join(Path(target_dir).parent, command_filename),
                                        to_print=True
                                        )

    # read file
    wb = pd.ExcelFile(src_path)

    df_dict = pd.read_excel(wb, sheet_name=None)
    df_dict.pop('Status list', None)

    # Word separator for tokenized text - default
    token_sep = ' '
    column_save_sep = ','

    df_list = [
        MetaDataTools.field_descriptor_text_df_from_df(df=v, source=k,
                                                       is_labelled=True, sep=token_sep, to_tokenize=to_tokenize,
                                                       stemmer_name=stemmer_name)
        for k, v in df_dict.items()
        if k != 'Status list']
    tokenized_text = 'Tokenized' if to_tokenize else 'NotTokenized'
    stemmed_text = f'Stem{stemmer_name}' if stemmer_name != 'None' else 'NotStemmed'

    base_save_name = f'{output_type}_{tokenized_text}_{stemmed_text}'

    # If default output_type (i.e. 'tokenized') then leave as is.
    df = MetaDataTools.collate_dfs_from_list(df_list=df_list)

    text_columns = ['Source', 'Field', 'Descriptor']
    if to_tokenize:
        text_columns = ['Tokenized Source', 'Field(lower)', 'Tokenized Descriptor']
    df = MetaDataTools.prep_df_for_bert_all(df=df, text_columns=text_columns)

    # Save:
    # - labels/categories
    # - dataset with labels
    # - dataset without labels
    save_name = f'Labelled_{base_save_name}'
    labels = MetaDataTools.save_unique_labels_from_df(
        df=df, label_column='category', save_name=save_name, save_dir=target_dir, prefix=prefix)

    # Split into training and test data, in ratio 9:1, with examples of each label
    main_dfs = []
    test_dfs = []
    # Shuffles the data, use random_state for reproducibility of result
    df = df.sample(frac=1, random_state=5465)
    for label in labels:
        df_by_label = df.loc[df['category'] == label]
        test_len = int(round(len(df_by_label) / 10 + 0.5))  # Round up to integer
        df_label_test = df_by_label.head(test_len)
        df_label_main = df_by_label.tail(len(df_by_label) - test_len)
        main_dfs.append(df_label_main)
        test_dfs.append(df_label_test)

    df_main = pd.concat(main_dfs)
    df_test = pd.concat(test_dfs)

    # Re-shuffle df_main, and set order of df_test
    df_main.sample(frac=1)
    df_test = df_test.sort_values(['Source', 'Label'])

    MetaDataTools.save_df(
        df=df_main, save_name=f'Labelled_{base_save_name}_main.csv', save_dir=target_dir, prefix=prefix,
        sep=column_save_sep)
    MetaDataTools.save_df(
        df=df_test, save_name=f'Labelled_{base_save_name}_test.csv', save_dir=target_dir, prefix=prefix,
        sep=column_save_sep)

    columns = [column for column in df.columns if str(column) != 'category']
    df = df[columns]
    MetaDataTools.save_df(
        df=df, save_name=f'NotLabelled_{base_save_name}.csv', save_dir=target_dir, prefix=prefix, sep=column_save_sep)


if __name__ == '__main__':
    main()

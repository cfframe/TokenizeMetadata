# tokenize_meta_data.py
import argparse
import datetime

from src.file_tools import FileTools
from src.meta_data_tools import MetaDataTools
import os
from pathlib import Path

TEST_FILEPATH = os.path.join(Path(__file__).parent, 'test', 'test_data', 'test_tsv_5_cols_inc_labels.txt')

"""
Description: Read in a data dictionary file and output a tokenized version.

Example usage:
py tokenize_meta_data.py -s "C:/temp/TableMetaData/Source/SAP IS-H Case Attribute.txt" -td C:/temp/TableMetaData/Results 
py tokenize_meta_data.py -s "C:/temp/TableMetaData/Source" -d -td C:/temp/TableMetaData/Results2 --to_tokenize
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Extract fields and tokenized descriptors from TSV files.')
    src_help = f'Source path for processing. Assume a file, but use is_directory flag if a folder of files. ' \
               f'Default: {TEST_FILEPATH}'
    parser.add_argument('-s', '--src_path', type=str, default=TEST_FILEPATH,
                        help=src_help)
    parser.add_argument('-d', '--is_directory', action='store_true',
                        help='When used, indicates src_path is a directory and not a file.')
    parser.add_argument('-td', '--target_dir', type=str, default=Path(__file__).parent,
                        help='Working directory for saving files etc. Default: parent directory of this script.')
    parser.add_argument('-tt', '--to_tokenize', action='store_true',
                        help='When used, indicates that descriptor text is to be tokenized.')
    parser.add_argument('-sl', '--save_label', type=str, default='',
                        help='Optional label to add to save name for easier identification.')
    parser.add_argument('-ss', '--save_suffix', type=str, default='.csv',
                        help='Suffix/extension for saving files. Default: ".csv"')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # Declare args - helps with auto-completion. Convert to object?
    src_path = args.src_path
    target_dir = args.target_dir
    is_directory = args.is_directory
    save_label = args.save_label
    save_suffix = args.save_suffix
    to_tokenize = args.to_tokenize

    if len(save_label) == 0:
        save_label = Path(src_path).stem[:3]

    if len(str(target_dir)) == 0 or Path(target_dir) == Path(__file__).parent:
        print('No directory supplied, so exit program.')
        quit()

    if input(f'WARNING Will clear directory {target_dir} if it exists.\n'
             f'Continue (n and Enter, or just Enter to continue)?').lower() == 'n':
        print('Chosen to quit.')
        quit()

    FileTools.ensure_empty_directory(target_dir)

    prefix = f'{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}{save_label}'
    command_filename = f'{prefix} Command args.txt'
    FileTools.save_command_args_to_file(vars(args),
                                        save_path=os.path.join(Path(target_dir).parent, command_filename),
                                        to_print=True
                                        )
    if not is_directory:
        MetaDataTools.field_descriptors_df_from_file(
            src_path, target_dir, prefix, to_save=True, to_tokenize=to_tokenize)
    else:
        df_list, errors = \
            MetaDataTools.list_of_field_descriptors_dfs_from_files(
                src_path, target_dir, prefix, to_save=True, to_tokenize=to_tokenize)

        for err in errors:
            print(err)

        save_name = f'collated{save_suffix}'

        collated_dfs = MetaDataTools.collate_dfs_from_list(df_list=df_list)

        if len(collated_dfs) > 0:
            print('First few records in collated DataFrames:\n')
            print(collated_dfs.head())

        MetaDataTools.save_df(df=collated_dfs, save_name=save_name, save_dir=target_dir, prefix=prefix)


if __name__ == '__main__':
    main()

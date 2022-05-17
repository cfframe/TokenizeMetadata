# test_meta_data_tools.py
from pathlib import Path

import numpy as np

from src.file_tools import FileTools
import nltk
import pandas as pd
import os
import unittest
from src.meta_data_tools import MetaDataTools as MDT
from src.custom_exceptions import DataFrameException


class MetaDataToolsTestCase(unittest.TestCase):
    def setUp(self):
        """Fixtures used by test."""
        self.Root = Path(__file__).parent
        self.Temp = os.path.join(self.Root, 'temp_meta_data_tools')
        self.TestDataDir = os.path.join(self.Root, 'test_data')
        self.TestExcelFilesDir = os.path.join(self.Root, 'test_excel_files')
        self.ErrorCheckDir = os.path.join(self.Root, 'error_check')
        self.Test2ColFile = os.path.join(self.TestDataDir, 'test_tsv_2_cols.txt')
        self.Test5ColIncLabelDataFrame = MDT.read_raw_data(os.path.join(self.TestDataDir, 'test_tsv_5_cols_inc_labels.txt'))
        self.TestTokenizedLabelledDataFrame = MDT.read_raw_data(os.path.join(self.TestDataDir, 'test_tokenized_and_labelled.txt'))
        self.TestNoDescDataFrame = MDT.read_raw_data(os.path.join(self.TestDataDir, 'test_no_desc_tsv.txt'))
        self.Test1ColDataFrame = MDT.read_raw_data(os.path.join(self.TestDataDir, 'test_tsv_1_col.txt'))
        self.Test2ColDataFrame = MDT.read_raw_data(self.Test2ColFile)
        self.ExcelXlsxFilePath = os.path.join(self.TestExcelFilesDir, 'test_xls.xlsx')
        self.ExcelXlsmFilePath = os.path.join(self.TestExcelFilesDir, 'test_xlsm.xlsm')

        FileTools.ensure_empty_directory(self.Temp)

    def tearDown(self) -> None:
        FileTools.ensure_empty_directory(self.Temp)

    def test_read_raw_data__raw_data_has_expected_shape(self):
        df = self.Test5ColIncLabelDataFrame

        pd.set_option('display.max_colwidth', 100)
        print(df.head())
        self.assertEqual((7, 5), df.shape)

    def test_process_text_in_dataframe__returns_dataframe_with_text_stemmed(self):
        test_df = self.Test5ColIncLabelDataFrame
        columns_to_lower = [0, 4]
        columns_to_tokenize = [1]
        new_df = MDT.process_text_in_dataframe(test_df, columns_to_lower, columns_to_tokenize,
                                               stemmer_name='lancaster')

        with self.subTest(self):
            expected = 'institut'
            actual = new_df['Tokenized Some Description'][0]
            self.assertEqual(expected, actual)

        with self.subTest(self):
            expected = 'h fred cas typ'
            actual = new_df['Tokenized Some Description'][1]
            self.assertEqual(expected, actual)

    def test_process_text_in_dataframe__returns_dataframe_with_punctuation_stripped(self):
        test_df = self.Test5ColIncLabelDataFrame
        columns_to_lower = [0, 4]
        columns_to_tokenize = [1]
        new_df = MDT.process_text_in_dataframe(test_df, columns_to_lower, columns_to_tokenize)

        expected = 'h,fred,case,type'
        actual = new_df['Tokenized Some Description'][1]
        self.assertEqual(expected, actual)

    def test_process_text_in_dataframe__returns_dataframe_without_tokenized(self):
        test_df = self.Test5ColIncLabelDataFrame
        columns_to_lower = [0, 4]
        columns_to_tokenize = [1]
        new_df = MDT.process_text_in_dataframe(test_df, columns_to_lower, columns_to_tokenize,
                                               to_tokenize=False)

        with self.subTest(self):
            print('Test that an original field remains')
            self.assertTrue('Some Description' in new_df.columns)

        with self.subTest(self):
            print('Test that tokenized version does not exist')
            self.assertFalse('Tokenized Some Description' in new_df.columns)

    def test_cleanse_text__remove_specified_punctuation(self):
        expected = 'dogs kennels paint'
        with self.subTest(self):
            print(f'Testing for: Apostrophe')
            test_text = "Dog's kennel's paint"
            actual = ' '.join(MDT.cleanse_text(test_text))
            self.assertEqual(expected, actual)

        with self.subTest(self):
            print(f'Testing for: Double quote')
            test_text = 'Dogs "kennels" paint'
            actual = ' '.join(MDT.cleanse_text(test_text))
            self.assertEqual(expected, actual)

    def test_cleanse_text__replace_other_punctuation(self):
        expected = 'dogs kennel dog house'.split(' ')
        sub_tests = [['Hyphen', "Dog's kennel dog-house"],
                     ['Equals sign', "dog's kennel=dog house"],
                     ['Adjacent punctuation', "Dog's kennel=dog****house&&&"]]

        for sub_test in sub_tests:
            with self.subTest(self):
                print(f'Testing for: {sub_test[0]}')
                test_text = sub_test[1]
                actual = MDT.cleanse_text(test_text)
                self.assertEqual(expected, actual)

    def test_stemmer__by_stemmer(self):
        test_words = ['describes', 'describe', 'descriptor', 'description']
        sub_tests = [['Porter', nltk.PorterStemmer(), ['describ', 'describ', 'descriptor', 'descript']],
                     ['Lancaster', nltk.LancasterStemmer(), ['describ', 'describ', 'describ', 'describ']]]
        for sub_test in sub_tests:
            with self.subTest(self):
                print(f'Testing for: {sub_test[0]}')
                stemmer = sub_test[1]
                expected = sub_test[2]
                actual = MDT.stemming(test_words, stemmer)
                self.assertEqual(expected, actual)

    def test_identify_descriptor_column__when_valid_column_name_exists__returns_index_and_name(self):
        df = self.Test5ColIncLabelDataFrame
        expected = 1
        # Column index is first part of pairing returned
        actual = MDT.identify_descriptor_column(self.Test5ColIncLabelDataFrame)[0]

        self.assertEqual(expected, actual)

    def test_identify_descriptor_column__when_valid_column_name_not_exists__returns_dummy_index_and_name(self):
        expected = -1
        # Column index is first part of pairing returned
        actual = MDT.identify_descriptor_column(self.TestNoDescDataFrame)[0]

        self.assertEqual(expected, actual)

    def test_field_tokenized_descriptor_list_from_df__when_valid__returns_paired_series_list(self):
        # Column index is first part of pairing returned
        sub_tests = [['2 columns', self.Test2ColDataFrame], ['5 columns', self.Test5ColIncLabelDataFrame]]
        for sub_test in sub_tests:
            with self.subTest(self):
                print(f'Testing for: {sub_test[0]}')
                fdl = MDT.field_tokenized_descriptor_list_from_df(sub_test[1])

                self.assertTrue(len(fdl) == 2)
                self.assertTrue(fdl[0].__class__.__name__ == 'Series')
                self.assertTrue(fdl[1].__class__.__name__ == 'list')

    def test_field_tokenized_descriptor_list_from_df__when_invalid__raises_exception(self):
        # Column index is first part of pairing returned
        sub_tests = [['1 column', self.Test1ColDataFrame], ['No descriptor', self.TestNoDescDataFrame]]
        for sub_test in sub_tests:
            with self.subTest(self):
                print(f'Testing for: {sub_test[0]}')
                self.assertRaises(DataFrameException, MDT.field_tokenized_descriptor_list_from_df, sub_test[1])

    def test_field_tokenized_descriptor_list_from_df__last_element_is_expected_type(self):
        sub_tests = [['treat as not labelled', False, 'list'], ['treat as labelled', True, 'Series']]
        df = self.Test5ColIncLabelDataFrame
        for sub_test in sub_tests:
            testing_for = sub_test[0]
            is_labelled = sub_test[1]
            expected = sub_test[2]

            print(f'Testing for: {testing_for}')
            result = MDT.field_tokenized_descriptor_list_from_df(df, is_labelled)

            self.assertEqual(expected, type(result[-1]).__name__)

    def test_field_tokenized_descriptor_list_from_df__when_valid__tokenizes_data(self):
        fdl = MDT.field_tokenized_descriptor_list_from_df(self.Test2ColDataFrame)
        expected = 'institution'
        actual = fdl[1][0]
        self.assertEqual(expected, actual)

    def test_field_descriptor_text_df_from_df__when_to_tokenize__includes_tokenized(self):
        fdl = MDT.field_descriptor_text_df_from_df(self.Test2ColDataFrame, 'test_name', to_tokenize=True)

        with self.subTest(self):
            print('Tokenized, verify tokenized columns exist')
            tokenized_columns = [x for x in fdl.columns if str(x).startswith('Tokenized')]
            self.assertTrue(len(tokenized_columns) > 0)

        with self.subTest(self):
            print('Tokenized, verify contains expected text')
            expected = 'h fred case type'
            actual = fdl['Tokenized Descriptor'][1]
            self.assertEqual(expected, actual)

    def test_field_descriptor_text_df_from_df__when_not_to_tokenize__has_no_tokenized(self):
        fdl = MDT.field_descriptor_text_df_from_df(self.Test2ColDataFrame, 'test_name', to_tokenize=False)

        tokenized_columns = [x for x in fdl.columns if str(x).startswith('Tokenized')]
        expected = 'IS H   Fred  Case Type'
        actual = fdl['Descriptor'][1]

        with self.subTest(self):
            print('Not tokenized but still contains expected text')
            self.assertEqual(expected, actual)

        with self.subTest(self):
            print('Not tokenized, verify no tokenized columns')
            self.assertTrue(len(tokenized_columns) == 0)

    def test_field_descriptor_text_df_from_df__when_labelled__includes_label_column(self):
        fdl = MDT.field_descriptor_text_df_from_df(self.Test5ColIncLabelDataFrame, 'test_name', is_labelled=True)

        self.assertTrue('Label' in fdl.columns)

    def test_field_descriptors_df_from_file__when_valid_file(self):
        src_path = self.Test2ColFile
        target_dir = self.Temp
        prefix = 'dummy'

        with self.subTest(self):
            print('Testing for: empty directory pre-test')
            expected = 0
            actual = len([name for name in os.listdir(self.Temp) if os.path.isfile(os.path.join(self.Temp, name))])
            self.assertEqual(expected, actual)

        result = MDT.field_descriptors_df_from_file(src_path, target_dir, prefix, to_save=True)

        with self.subTest(self):
            print('Testing for: return DataFrame')
            expected = 'DataFrame'
            actual = result.__class__.__name__
            self.assertEqual(expected, actual)

        with self.subTest(self):
            print('Testing for: save processed file')
            expected = 1
            actual = len([name for name in os.listdir(self.Temp) if os.path.isfile(os.path.join(self.Temp, name))])
            self.assertTrue(expected, actual)

    def test_dict_of_field_descriptors_dfs_from_files(self):
        src_path = self.ErrorCheckDir
        target_dir = self.Temp
        prefix = 'dummy'

        dataframes, errors = MDT.dict_of_field_descriptors_dfs_from_files(src_path, target_dir, prefix, to_save=True)

        with self.subTest():
            testing_for = 'Expected number of returned DataFrames'
            print(f'Testing for: {testing_for}')
            expected = 2
            actual = len(dataframes)

            self.assertEqual(expected, actual)

        with self.subTest():
            testing_for = 'Expected number of errors'
            print(f'Testing for: {testing_for}')
            # expected = len([name for name in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, name))])
            expected = 2
            actual = len(errors)

            self.assertEqual(expected, actual)

    def test_list_of_field_descriptors_dfs_from_files(self):
        src_path = self.ErrorCheckDir
        target_dir = self.Temp
        prefix = 'dummy'

        dataframes, errors = MDT.list_of_field_descriptors_dfs_from_files(src_path, target_dir, prefix, to_save=True)

        with self.subTest():
            testing_for = 'Expected number of returned DataFrames'
            print(f'Testing for: {testing_for}')
            expected = 2
            actual = len(dataframes)

            self.assertEqual(expected, actual)

        with self.subTest():
            testing_for = 'Expected number of errors'
            print(f'Testing for: {testing_for}')
            # expected = len([name for name in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, name))])
            expected = 2
            actual = len(errors)

            self.assertEqual(expected, actual)

    def test_collate_dfs_from_list(self):
        dataframes, errors = MDT.list_of_field_descriptors_dfs_from_files(
            src_path=self.TestDataDir, target_dir=self.Temp, prefix='from_list', to_save=False)

        df = MDT.collate_dfs_from_list(df_list=dataframes)
        expected_columns = ['Source', 'Field', 'Field(lower)', 'Tokenized Source', 'Descriptor', 'Tokenized Descriptor']
        compare_column_names = (df.columns == expected_columns)

        self.assertTrue(compare_column_names.all())

    def test_save_df(self):
        dataframes, errors = MDT.list_of_field_descriptors_dfs_from_files(
            src_path=self.TestDataDir, target_dir=self.Temp, prefix='from_list', to_save=False)

        df = MDT.collate_dfs_from_list(df_list=dataframes)

        MDT.save_df(df=df, save_dir=self.Temp, save_name='all_data.txt', prefix='from_list_')
        self.assertTrue(Path(os.path.join(self.Temp, 'from_list_all_data.txt')).is_file())

    def test_prep_df_for_bert__tokenized(self):
        df_all = MDT.field_descriptor_text_df_from_df(
            self.TestTokenizedLabelledDataFrame, 'test_name', is_labelled=True, sep=' ', to_tokenize=True)

        df = MDT.prep_df_for_bert(df_all)

        with self.subTest():
            testing_for = 'prep_df_for_bert expected columns'
            print(f'Testing for: {testing_for}')
            expected_columns = ['category', 'text']
            self.assertTrue((expected_columns == df.columns).all())

        with self.subTest():
            testing_for = 'prep_df_for_bert no commas'
            print(f'Testing for: {testing_for}')
            for column in df.columns:
                self.assertFalse(df[column].str.contains(',').any())

        df = MDT.prep_df_for_bert(df_all, text_columns=['Source', 'Field', 'Descriptor'])

        with self.subTest():
            testing_for = 'prep_df_for_bert raw text'
            print(f'Testing for: {testing_for}')
            for column in df.columns:
                self.assertFalse(df[column].str.contains(',').any())

        with self.subTest():
            testing_for = 'prep_df_for_bert input has no_label rows removed'
            print(f'Testing for: {testing_for}')
            no_label_count = len(df_all[df_all['Label'] == 'no_label'])
            self.assertTrue(no_label_count > 0)
            no_label_count = len(df[df['category'] == 'no_label'])
            self.assertTrue(no_label_count == 0)

    def test_prep_df_for_bert_all__tokenized(self):
        df_all = MDT.field_descriptor_text_df_from_df(
            self.TestTokenizedLabelledDataFrame, 'test_name', is_labelled=True, sep=' ', to_tokenize=True)

        df = MDT.prep_df_for_bert_all(df_all)

        with self.subTest():
            testing_for = 'prep_df_for_bert_all keeps original columns'
            print(f'Testing for: {testing_for}')
            expected_columns = ['Source',
                                'Field',
                                'Field(lower)',
                                'Tokenized Source',
                                'Descriptor',
                                'Tokenized Descriptor',
                                'Label',
                                'category',
                                'text']

            self.assertTrue((expected_columns == df.columns).all())

        with self.subTest():
            testing_for = 'prep_df_for_bert_all no commas'
            print(f'Testing for: {testing_for}')
            for column in [x for x in df.columns if str(x) in ['category', 'text']]:
                self.assertFalse(df[column].str.contains(',').any())

        df = MDT.prep_df_for_bert_all(df_all, text_columns=['Source', 'Field', 'Descriptor'])

        with self.subTest():
            testing_for = 'prep_df_for_bert_all raw text'
            print(f'Testing for: {testing_for}')
            for column in [x for x in df.columns if str(x) in ['category', 'text']]:
                self.assertFalse(df[column].str.contains(',').any())

        with self.subTest():
            testing_for = 'prep_df_for_bert_all input has no_label rows removed'
            print(f'Testing for: {testing_for}')
            no_label_count = len(df_all[df_all['Label'] == 'no_label'])
            self.assertTrue(no_label_count > 0)
            no_label_count = len(df[df['category'] == 'no_label'])
            self.assertTrue(no_label_count == 0)

    def test_save_unique_labels_from_df(self):
        df = self.Test5ColIncLabelDataFrame
        actual = MDT.save_unique_labels_from_df(df=df, label_column='Label', save_dir=self.Temp, save_name='SaveName', prefix='Prefix')

        with self.subTest(self):
            print('Testing for: expected array')
            expected = np.array(['key', 'object'])
            self.assertTrue(np.array_equal(expected, actual))

        with self.subTest(self):
            print('Testing for: save processed file')
            expected = 1
            actual = len([name for name in os.listdir(self.Temp) if os.path.isfile(os.path.join(self.Temp, name))])
            self.assertTrue(expected, actual)


if __name__ == '__main__':
    unittest.main()

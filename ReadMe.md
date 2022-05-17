# ReadMe for TokenizeMetadata
## Overview
The relates to WP2, extraction of metadata. Each script attempts to extract metadata from a data dictionary
and produces data in a format suitable for MetadataClassifier training. The intention is to use that data
as the input to a system which can learn a significant portion of the rules for classifying fields for the 
Smart Patient Health Record structure, with a view to ultimately assisting a clinician with that task.

The name comes form initial development having been targeted at producing tokenized data. Full tokenizing is now 
optional.  All data goes through some cleansing, to remove punctuation. Fuller tokenization currently also:
- makes all words lower case
- removes stopwords (English only at this point)

Script tokenize_labelled_meta_data.py in particular has an option for producing a file that can be used
as the input for BERT based model. BERT is a commonly used baseline for NLP experiments.


## General Python project basics
### Tools and technologies used:
<ul>
<li>PyCharm 2021.2.3</li>
<li>python 3.8.10 - packages as listed in requirements.txt</li>
</ul>

### Set up
Python package requirements are defined in requirements.txt. We used a virtual environment for installing these
to reduce the risk of package dependency issues.

Installation **can take a few minutes**, especially due to NLTK download.

One way of installing requirements: <br />
<code>py -m pip install --upgrade pip</code><br />
<code>pip install -r requirements.txt</code>

### Testing
Using unittest. Can run all test files with a single command in the Terminal window.

<code>
python -m unittest
</code>

Requirements for this
<ul>
<li>top level test directory to be named "test/"</li>
<li>empty file in test directory named "__init__.py"</li>
<li>test case file name format: "test_*.py"</li>
<li>tests encapsulated in unittest.TestCase classes</li>
</ul>

## Usage
Primary scripts:
<ul>
<li><i>tokenize_labelled_meta_data.py</i></li>
<li><i>tokenize_meta_data.py</i></li>
</ul>

Examples below assume they script is being run from the main application directory.

Help text available by running <code>py <i>script_name</i>.py -h</code> in the terminal.

### tokenize_labelled_meta_data.py
Extract fields, tokenized descriptors and labels from data dictionary (Excel workbook) and output a tokenized version 
as a text file.

Primary usage is to produce output suitable for further training of a BERT model.

Expected format of the input Excel workbook:
- Worksheet 'Status list' (optional). This is ignored in processing.
- All other worksheets are suitably named and have rows with these columns:
  - first column: field names
  - a subsequent column: descriptors (with a name similar to 'description' or 'descriptor')
  - last column: labels
  - all other columns are ignored.
  The worksheet name is taken as the 'Source' (source table name) of the metadata. 

Outputs several files:
- *_test.csv - a random selection of about 10% of the processed rows, excluding items labelled as 'no_label';
includes new fields 'category' and 'text'. To be used for final testing of a trained model.
- *_main.csv - the remaining processed data, to be used for training and validation, same profile as test data
- *NotLabelled*.csv - full set of data, but without 'category'; if the original set has a 'Label' column, this will be 
retained
- Labels*.txt - a list of labels found in the supplied data

Parameters:

<code>-s</code>, <code>--src_path</code>: Source path for processing. Default: TEST_FILEPATH, as defined in tokenize_labelled_meta_data.py.  

<code>-td</code>, <code>--target_dir</code>: Working directory for saving files etc. Default: parent directory of tokenize_labelled_meta_data.py.

<code>-ot</code>, <code>--output_type</code>: Either "tsv", or "bert". In both cases, the "source" is taken as the worksheet name; 
field and descriptor columns are identified and extracted. Options:
<ul>
<li>"tsv" - produces a text file with tsv separated column values</li>
<li>"bert" - outputs a simple csv file, each record being of the format <i>category</i>,<i>text</i></li>
</ul>
<code>-tt</code>, <code>--to_tokenize</code>: Whether text is tokenized.
<code>-sn</code>, <code>--stemmer_name</code>: Name of NLTK stemmer (default: none).
<code>-sl</code>, <code>--save_label</code>: Optional label to add to save name for easier identification.

Examples:
```commandline
python tokenize_labelled_meta_data.py -s "C:/temp/TableMetaData/Source/FCRB_Data Model_v0.5 CFF 1g.xlsm" -td C:/temp/TableMetaData/Results
python tokenize_labelled_meta_data.py -s "data/HIC metadata.xlsx" -td C:/temp/TableMetaData/Results -ot bert
```

### tokenize_meta_data.py
Extract fields, labels and tokenized descriptors from a TSV file or an Excel file.

Does not include stemming at this point.

Parameters:

<code>-s</code>, <code>--src_path</code>: Source path for processing. Assume a file, but use is_directory flag if a folder. 
Default: TEST_FILEPATH, as defined in tokenize_meta_data.py.  

<code>-d</code>, <code>--is_directory</code>: Indicates src_path is a directory and not a file. Default: true

<code>-td</code>, <code>--target_dir</code>: Working directory for saving files etc. Default: parent directory of tokenize_meta_data.py.

<code>-ext</code>, <code>--suffix</code>: Suffix/extension for saving files.

Example:<br />
```commandline 
python tokenize_meta_data.py -s "C:/temp/TableMetaData/Source/SAP IS-H Case Attribute.txt" -td C:/temp/TableMetaData/Results
```


import os
import csv
import pandas as pd

unnumbered_sections = ['references', 'acknowledgement', 'acknowledgements', 'reproducibility statement', 'reproducibility', 'limitations', 'ethics statement', 'acknowledgements and disclosure of funding', 'statement of reproducibility', 'reproducibility & ethics statements', 'acknowledgments', 'eferences', 'ethics and reproducibility']

metadata_sections_order = ['title', 'abstract']
other_sections_order = ['introduction', 'related work', 'methodology', 'experiments', 'results', 'discussion', 'conclusion', 'future work']


def read_csv_to_dict(freq_file):
    assert os.path.exists(freq_file)
    with open(freq_file, 'r') as fp:
        reader = csv.reader(fp)
        global_word_frequencies = {row[0]:int(row[1]) for row in reader}
    return global_word_frequencies


def combine_same_sections(section_content_list):
    #print('section_content_list: %s' % section_content_list)
    combined = ''
    for sc in section_content_list:
        #print('sc: %s' % sc)
        if not isinstance(sc, float) and not sc != sc:
            if len(combined) > 0:
                combined += '\n'
            combined += sc
    return combined


def tidy_up_decisions(row):

    if row['decision'] in ['Accept (Oral)', 'Accept (Spotlight)', 'Accept (Poster)', 'Reject']:
        return row['decision']
    elif row['binary_decision'] == 1:
        return 'Accept'
    elif row['binary_decision'] == 0:
        return 'Reject'
    else:
        assert False
    

def one_row_per_paper(paperDF):

    # Check the second paper id column is redundant
    #assert paperDF['paper_id'].equals(paperDF['Paper ID'])
    #paperDF = paperDF.drop(columns=['Paper ID'])

    #paperDF = paperDF.rename(columns={'Section Name': 'section_name', 'Section Content': 'section_content'})

    paperDF['section_tuple'] = paperDF.apply(lambda x: (x['section_name'], x['section_content']), axis=1)

    # Updated to reflect new sections
    #combinedDF = paperDF.groupby(['paper_id', 'decision', 'title', 'abstract'])['section_tuple'].apply(list).reset_index(name='section_tuple')
    combinedDF = paperDF.groupby(['paper_id'])['section_tuple'].apply(list).reset_index(name='section_tuple')
    
    if any(combinedDF['paper_id'].duplicated()):
        print('*** WARNING: DUPLICATES ARE PRESENT IN paper_id ***')
        combinedDF = combinedDF.drop_duplicates('paper_id', keep='first')
        
    return combinedDF


def xlsx_details(xlsx_file):
    fullDF = pd.read_excel(xlsx_file, sheet_name='Sheet6')
    fullDF = fullDF.drop_duplicates()
    print('xlsx sheet6 entries: %d' % fullDF.shape[0])
    #print('xlsx sheet6 columns: %s' % fullDF.columns)
    tempDF = pd.read_excel(xlsx_file, sheet_name='Sheet5')
    tempDF = tempDF.drop_duplicates()
    assert not 'decision123' in tempDF.columns
    # Drop any rows with no decision
    tempDF = tempDF.dropna(subset=['decision'])
    tempDF['decision123'] = tempDF['decision'].apply(lambda x: 'Accept' if x == 1 else 'Reject')
    print('xlsx sheet5 entries: %d' % tempDF.shape[0])
    #print('xlsx sheet5 columns: %s' % tempDF.columns)
    sheet6_ids = fullDF['paper_id'].tolist()
    sheet5_ids = tempDF['paper_id'].tolist()
    assert len(list(set(sheet5_ids) & set(sheet6_ids))) == 0
    # Join up the two sheets (known to have no duplicate paper_ids)
    fullDF = pd.concat([fullDF, tempDF])
    fullDF = fullDF.drop(columns=['id', 'citenum'])
    print('fullDF entries: %d' % fullDF.shape[0])
    # Perform column renames
    assert 'decision123' in fullDF.columns
    fullDF = fullDF.drop(columns=['title'])
    fullDF = fullDF.rename(columns={'decision': 'binary_decision', 'decision123': 'decision', 'Title1': 'title'})
    # Construct or tidy word level decisions
    fullDF['decision'] = fullDF.apply(lambda x: tidy_up_decisions(x), axis=1)
    print('fullDF columns: %s' % fullDF.columns)
    return fullDF


def combine_all_info(merged_file, equivalent_section_file, xlsx_file):

    # Read in all the merged information
    paperDF = pd.read_csv(merged_file)
    print('Num rows in paperDF: %d' % paperDF.shape[0])
    print('paperDF columns: %s' % paperDF.columns)

    # Standardize column names
    if 'section_name_grobid' in paperDF.columns:
        paperDF = paperDF.rename(columns={'section_name_grobid': 'section_name', 'section_content_vila': 'section_content'})
    if 'section_name_vila' in paperDF.columns:
        paperDF = paperDF.rename(columns={'section_name_vila': 'section_name', 'section_content_vila': 'section_content'})
    if 'section_header_offsets' in paperDF.columns:
        paperDF = paperDF.drop(columns=['section_header_offsets', 'subheadings_grobid'])

    # Drop all columns we don't want any more
    paperDF = paperDF.drop(columns=paperDF.columns.difference(['paper_id', 'section_name', 'section_content']))
    print('Remaining columns: %s' % paperDF.columns)

    # Map section names
    equivalentsDF = pd.read_csv(equivalent_section_file)
    equivalentsDict = dict(zip(equivalentsDF.equivalent_section, equivalentsDF.main_section))

    # Discard any null values
    print('Num rows before null value removal: %d' % paperDF.shape[0])
    paperDF = paperDF.dropna()
    print('Num rows after null value removal: %d' % paperDF.shape[0])
    
    paperDF['section_name'] = paperDF['section_name'].apply(lambda x: x.lower().strip())
    paperDF['section_name'] = paperDF['section_name'].apply(lambda x: equivalentsDict[x] if x in equivalentsDict else x)

    # Could drop any undesirable section names here (e.g. abstract)
        
    # Aggregate content for the same section names
    paperDF = paperDF.groupby(['paper_id', 'section_name'], as_index=False)['section_content'].apply(lambda x: combine_same_sections(x)).reset_index()
    
    # Reorganize data so that each paper is only listed once
    paperDF = one_row_per_paper(paperDF)
    print('Number of one row papers: %d' % paperDF.shape[0])

    ## Read in review, title and abstract details
    fullDF = xlsx_details(xlsx_file)
    
    missing_paper_ids = [x for x in paperDF['paper_id'].tolist() if not x in fullDF['paper_id'].tolist()]
    if len(missing_paper_ids) > 0:
        print('Sample of missing papers: %s' % missing_paper_ids[:10])
    
    ## Merge section information and the decision / review data
    # Add in ratings and reviews
    num_papers = paperDF.shape[0]
    paperDF = pd.merge(paperDF, fullDF, on='paper_id', how='inner')
    print('Original number of papers: %d' % num_papers)
    print('Merging paperDF and fullDF results in: %d papers' % paperDF.shape[0])
    #assert paperDF.shape[0] == num_papers
    
    return paperDF


def rearrange_annotations(statementDF):

    completeDF = statementDF[['paper_id', 'sentiment']].drop_duplicates().groupby('paper_id', as_index=False).agg(list)
    completeDF['sentiment'] = completeDF['sentiment'].apply(lambda x: sorted(x))
    # Reduce to ones that have at least two
    wanted_ids = completeDF[completeDF['sentiment'].isin([['neg', 'pos'], ['neg', 'other', 'pos']])]['paper_id'].unique()
    print('Number of wanted ids: %d' % len(wanted_ids))
    completeDF = statementDF[statementDF['paper_id'].isin(wanted_ids)].groupby(['paper_id', 'sentiment'])['sentence'].apply(list).reset_index(name='sentences')
    
    # Rearrange columns
    entry = {}
    data = []
    for index, row in completeDF.iterrows():
        if not row['paper_id'] in entry:
            if len(entry) > 0:
                key = list(entry.keys())[0]
                if not 'other' in entry[key]:
                    data.append({'paper_id': key, 'pos': entry[key]['pos'], 'neg': entry[key]['neg'], 'other': []})
                else:
                    data.append({'paper_id': key, 'pos': entry[key]['pos'], 'neg': entry[key]['neg'], 'other': entry[key]['other']})
                entry = {}
            entry[row['paper_id']] = {}
        entry[row['paper_id']][row['sentiment']] = row['sentences']

    shotDF = pd.DataFrame.from_records(data)
    return shotDF


def filter_by_sections(completeDF, sections):

    if sections == ['all']:
        # No filtering is performed as all are being kept
        return completeDF
    
    #print('filter_by_sections: %s' % sections)
    completeDF['overlap'] = completeDF.apply(lambda x: len(list(set(flatten([y[0].split(' and ') for y in x['section_tuple']])) & set(sections))) == len(sections), axis=1)
    return completeDF[completeDF['overlap']].drop(columns=['overlap'])
        

def few_shot_statements(statement_file):
    statementDF = pd.read_csv(statement_file)
    print('Number of statements read: %d' % statementDF.shape[0])
    print('statementDF columns: %s' % statementDF.columns)
    statementDF = statementDF.drop_duplicates()
    print('Number of statements after de-duplication: %d' % statementDF.shape[0])

    completeDF = rearrange_annotations(statementDF)
    print('Number of complete rows: %d' % completeDF.shape[0])
    print('completeDF columns: %s' % completeDF.columns)

    return completeDF


def create_balanced_data(paperDF, num_shots=None, num_folds=None, test_only=False, decision_col='decision', random_state=234):

    # Extract paper_ids and associated result
    balancingDF = paperDF[['paper_id', decision_col]].drop_duplicates()
    print('counts of data to balance: %s' % balancingDF[decision_col].value_counts())

    valueDF = balancingDF[decision_col].value_counts().rename_axis(decision_col).reset_index(name='count')
    print('converted to DF: %s' % valueDF)

    if test_only:
        # The number of examples for each side
        if num_shots == 0:
            # Evaluating zero-shot on 20 examples only, i.e. 10 each side
            min_count = 10
        else:
            min_count = num_shots * num_folds
        assert min_count <= valueDF['count'].min()
    else:
        # If there isn't a limit on the number of instances, balance
        # to the lowest count
        min_count = valueDF['count'].min()
        
    print('Smallest count: %d' % min_count)

    # Added [0] to deal with case where multiples have same freq
    if len(valueDF[valueDF['count'] == valueDF['count'].min()][decision_col]) > 1:
        min_value = valueDF[valueDF['count'] == valueDF['count'].min()][decision_col][0].item()
    else:
        min_value = valueDF[valueDF['count'] == valueDF['count'].min()][decision_col].item()
    print('Value corresponding to smallest count: %s' % min_value)

    # NOTE: will have more 'Accept' examples with fine-grained decisions
    balancedDF = pd.DataFrame()
    for value in list(balancingDF[decision_col].unique()):
        balancedDF = pd.concat([balancedDF, balancingDF[balancingDF[decision_col] == value].sample(min_count, random_state=random_state)])
    print('balancedDF size: %d' % balancedDF.shape[0])
    print(balancedDF[decision_col].value_counts())
    print('balancedDF columns: %s' % balancedDF.columns)

    return paperDF[paperDF['paper_id'].isin(list(balancedDF['paper_id']))]


def order_list_by_another(smaller, bigger):
    assert set(smaller).issubset(bigger)
    return [x for x in bigger if x in smaller]


def divide_sections(sections):

    if sections == ['all']:
        return metadata_sections_order, sections
    
    metadata_sections = []
    other_sections = []
    for section in list(set(sections)):
        if section in ['abstract', 'title']:
            metadata_sections.append(section)
        else:
            other_sections.append(section)
    metadata_sections = order_list_by_another(metadata_sections, metadata_sections_order)
    other_sections = order_list_by_another(other_sections, other_sections_order)
    return metadata_sections, other_sections


def is_in_other_sections(combined_name, other_sections):
    for section in combined_name.split(' and '):
        if section in other_sections:
            return True
    return False


def create_combined_text(row, metadata_sections, other_sections):
    # Start with metadata_sections as these are columns (must be present)
    meta_texts = [row[x].strip() for x in metadata_sections if x in row]
    #print('meta_texts: %s' % meta_texts)
    # Process other_sections
    if other_sections == ['all']:
        other_texts = [x[1].strip() for x in row['section_tuple']]
    else:
        other_texts = [x[1].strip() for x in row['section_tuple'] if is_in_other_sections(x[0], other_sections)]
    #print('other_texts: %s' % other_texts)
    # Create combined text
    return ' '.join(meta_texts + other_texts)
            

def create_review_shots(completeDF, paperDF, section_list):

    # Add section details to completeDF
    #completeDF = pd.merge(completeDF, paperDF, how='inner', on='paper_id')
    print('Examples after addition of paperDF: %d' % completeDF.shape[0])
    print('Number of wanted paper ids: %d' % completeDF['paper_id'].nunique())
    print(completeDF.columns)
    
    print(completeDF['decision'].value_counts())
    print(completeDF['binary_decision'].value_counts())

    # Remove all training examples from rest
    paperDF = paperDF[~paperDF['paper_id'].isin(list(completeDF['paper_id']))]
    print('After removing completeDF paper_ids: %d' % paperDF.shape[0])
    
    # Reduce to maximal shots
    smallest_accept_class = min(completeDF[completeDF['decision'].str.startswith('Accept')]['decision'].value_counts())
    print('Smallest accept class: %s' % smallest_accept_class)

    
def flatten(xss):
    if xss == None:
        return []
    return [x for xs in xss for x in xs]


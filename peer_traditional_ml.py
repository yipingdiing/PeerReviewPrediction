import os
import csv
import spacy
import joblib
import argparse
import statistics
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import peer_useful
from scipy.sparse import lil_matrix

## Global file paths

pr_dir = os.path.join(os.environ['HOME'], "projects", "peer_review")
data_dir = os.path.join(pr_dir, "data")

# Equivalences of section names
equivalent_section_file = os.path.join(data_dir, "section_name_equivalents.csv")
assert os.path.exists(equivalent_section_file)

# Results file (if wanting to predict avg rating)
xlsx_file = os.path.join(os.environ['HOME'], "corpora", "peer_review", "openreview", "ICLR2021-2022", "data", "2017.xlsx")

# Peer info file
info_file = os.path.join(data_dir, "peer_info.csv.gz")
assert os.path.exists(info_file)
                         

def combine_all_text(row):
    meta_text = row['title'] + '\n' + row['abstract']
    other_text = '\n'.join([x[1] for x in row['section_tuple']])
    return meta_text + '\n' + other_text


def lemmatize(text):
    doc = nlp(text)
    # Turn it into tokens, ignoring the punctuation
    tokens = [token for token in doc if not token.is_punct]
    # Convert those tokens into lemmas, EXCEPT the pronouns, we'll keep those.
    lemmas = [token.lemma_ if token.pos_ != 'PRON' else token.orth_ for token in tokens]
    #print('Original: %s' % text)
    #print('Lemmatized: %s' % ' '.join(lemmas), flush=True)
    return ' '.join(lemmas)


if __name__ == '__main__':

    # Process command line arguments
    parser = argparse.ArgumentParser(description='Use traditional ML to predict acceptance decisions for papers.')
    parser.add_argument('--merged_data', type=str, default='mmda_sections.csv.gz', help='Name of extracted text file')
    parser.add_argument('--section', type=str, action='append', nargs='+', help='Section: title, abstract, introduction, methodology')
    parser.add_argument('--numeric_info', type=str, action='append', nargs='+', help='Numeric sections to use in prediction: page_count, citation_count, figure_count, table_count, section_count, reference_count')
    parser.add_argument('--kernel', type=str, default='rbf', help='Kernel options: linear, poly, rbf, sigmoid')
    parser.add_argument('--random_state', type=int, default=234, help='Random state: integer governing random actions')
    parser.add_argument('--num_folds', type=int, default=2, help='Number of folds to use')
    parser.add_argument('--min_count', type=int, default=2, help='min_count value for Doc2Vec')
    parser.add_argument('--vector_size', type=int, default=50, help='vector_size for Doc2Vec')
    parser.add_argument('--window', type=int, default=5, help='window for Doc2Vec')
    parser.add_argument('--epochs', type=int, default=40, help='epochs for Doc2Vec')
    parser.add_argument('--num_workers', type=int, default=10, help='num_workers for Doc2Vec')
    parser.add_argument('--algorithm', type=str, default='doc2vec', help='algorithm: doc2vec|tfidf|tf')
    parser.add_argument('--ngram_max', type=int, default=2, help='ngram maximum for TFIDF')
    parser.add_argument('--min_df', type=int, default=2, help='Ignore terms that have a document frequency strictly lower than the given threshold')
    parser.add_argument('--max_df', type=float, default=0.7, help='Ignore terms that have a document frequency strictly higher than the given threshold, if float in range [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts.')
    parser.add_argument('--output_file', type=str, default=os.path.join(data_dir, 'traditional_ml.csv'), help='Path to output CSV file')
    parser.add_argument('--lemmatizer', type=str, default=False, help='Lemmatizer on / off (True|False)')
    parser.add_argument('--model', type=str, default='SVC', help='model: SVC|DT|RF|MLP|AdaBoost')
    args = parser.parse_args()
    
    ## Process inputs
    merged_file = os.path.join(data_dir, args.merged_data)
    print('merged_file: %s' % merged_file)
    assert os.path.exists(merged_file)

    sections = list(set(peer_useful.flatten(args.section)))
    if 'all' in sections:
        # Remove any other individuals to prevent confusion
        sections = ['all']
    sections.sort()
    print('Paper sections being used: %s' % sections)

    ml_model = args.model
    print('ML model: %s (%s)' % (ml_model, type(ml_model)))
    if not ml_model.startswith('SVC-'):
        assert ml_model in ['DT', 'RF', 'MLP', 'AdaBoost']

    kernel = None
    if ml_model.startswith('SVC-'):
        divided = ml_model.split('-')
        ml_model = divided[0]
        kernel = divided[1]
        #kernel = args.kernel
        assert kernel in ['linear', 'poly', 'rbf', 'sigmoid']
    
    random_state = args.random_state
    num_folds = args.num_folds
    min_count = args.min_count
    vector_size = args.vector_size
    window = args.window
    epochs = args.epochs
    num_workers = args.num_workers
    algorithm = args.algorithm
    assert algorithm in ['doc2vec', 'tfidf', 'tf']
    min_df = args.min_df
    max_df = args.max_df
    ngram_max = args.ngram_max
    output_file = args.output_file
    lemmatizer = args.lemmatizer

    if lemmatizer == 'False':
        lemmatizer = False
    else:
        lemmatizer = True

    print('lemmatizer: %s' % str(lemmatizer))
        
    if algorithm != 'doc2vec':
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        print('Loaded spacy model: en_core_web_sm', flush=True)
    
    infoDF = pd.read_csv(info_file)
    numeric_info = peer_useful.flatten(args.numeric_info)
    for ni in numeric_info:
        assert ni in infoDF.columns
    print('Numeric information being used: %s' % numeric_info)
    # Drop all irrelevant infoDF columns
    infoDF = infoDF.drop(columns=infoDF.columns.difference(['paper_id'] + numeric_info))
    
    ## Gather up all data
    paperDF = peer_useful.combine_all_info(merged_file, equivalent_section_file, xlsx_file)

    ## Add in numeric info if needed
    if len(numeric_info) > 0:
        paperDF = pd.merge(paperDF, infoDF, on='paper_id', how='inner')
        # Drop any rows that don't have complete numeric info
        paperDF = paperDF.dropna(subset=numeric_info)
        print('Rows after addition of numeric columns: %d' % paperDF.shape[0])
    
    ## Balance data to create balanced train / test data
    balancedDF = peer_useful.create_balanced_data(paperDF, decision_col='binary_decision', random_state=random_state)
    
    # Extract annotations
    y = balancedDF["binary_decision"].tolist()
    
    ## Use Standard Scaler for the numeric data
    if len(numeric_info) > 0:
        X_numeric_scaled = StandardScaler().fit_transform(balancedDF[numeric_info])
        
    if len(sections) > 0:

        if algorithm == 'doc2vec':
        
            model_file = os.path.join(data_dir, "doc2vec." + args.merged_data.replace('.csv.gz', '') + ".v" + str(vector_size) + ".w" + str(window) + ".m" + str(min_count) + ".e" + str(epochs) + ".model")
            print('model_file: %s' % model_file)

            if not os.path.exists(model_file):

                print('Creating Doc2Vec model')
            
                ## Create Doc2Vec model from whole text collection
                paperDF['combined_text'] = paperDF.apply(lambda x: combine_all_text(x), axis=1)
                paperDF['combined_text'] = paperDF['combined_text'].apply(lambda x: x.replace('\n', ' '))

                # Tag all documents
                tagged_docs = [TaggedDocument(words=doc.split(), tags=[str(paper_id)]) for paper_id, doc in zip(paperDF['paper_id'].tolist(), paperDF['combined_text'].astype(str).tolist())]

                # Doc2Vec model
                model = Doc2Vec(tagged_docs, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, workers=num_workers, seed=random_state, report_delay=1.0)

                # Save model
                model.save(model_file)
                print('Saved doc2vec model to: %s' % model_file)

            ## Load model
            model = Doc2Vec.load(model_file)
            print('Loaded Doc2Vec model from: %s' % model_file)
        
            # Infer vectors for balanced data only
            metadata_sections, other_sections = peer_useful.divide_sections(sections)
            balancedDF['combined_text'] = balancedDF.apply(lambda x: peer_useful.create_combined_text(x, metadata_sections, other_sections), axis=1)
            balancedDF['combined_text'] = balancedDF['combined_text'].apply(lambda x: x.replace('\n', ' '))
            #print('combined_text: %s' % balancedDF.head(5)['combined_text'].tolist())
        
            X_text = [model.infer_vector(row['combined_text'].split()) for index, row in balancedDF.iterrows()]
            print('Inferred vectors for all balancedDFs: %d' % len(X_text))
            print(X_text[:5])

        else:
            assert algorithm == 'tfidf' or algorithm == 'tf'

            model_file = os.path.join(data_dir, algorithm + "." + args.merged_data.replace('.csv.gz', '') + "." + str(lemmatizer) + ".v" + str(vector_size) + ".n" + str(ngram_max) + ".min" + str(min_df) + ".max" + str(max_df).replace('.', '_') + ".pkl")
            print('model_file: %s' % model_file)

            if not os.path.exists(model_file):

                if lemmatizer:
                    if algorithm == 'tf':
                        tfidf_vectorizer = TfidfVectorizer(max_features=vector_size, ngram_range=(1, ngram_max), lowercase=True, token_pattern=r'[a-zA-Z][a-zA-Z]+', tokenizer=lemmatize, stop_words='english', min_df=min_df, max_df=max_df)
                    else:
                        assert algorithm == 'tfidf'
                        tfidf_vectorizer = TfidfVectorizer(max_features=vector_size, ngram_range=(1, ngram_max), lowercase=True, token_pattern=r'[a-zA-Z][a-zA-Z]+', tokenizer=lemmatize, stop_words='english', min_df=min_df, max_df=max_df, use_idf=False)
                        
                else:
                    if algorithm == 'tf':
                        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, ngram_max), lowercase=True, token_pattern=r'[a-zA-Z][a-zA-Z][a-zA-Z]+', stop_words='english', min_df=min_df, max_df=max_df)
                    else:
                        assert algorithm == 'tfidf'
                        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, ngram_max), lowercase=True, token_pattern=r'[a-zA-Z][a-zA-Z][a-zA-Z]+', stop_words='english', min_df=min_df, max_df=max_df, use_idf=False)

                ## Create TF*IDF model from whole text collection
                paperDF['combined_text'] = paperDF.apply(lambda x: combine_all_text(x), axis=1)
                paperDF['combined_text'] = paperDF['combined_text'].apply(lambda x: x.replace('\n', ' '))
                x_train = tfidf_vectorizer.fit_transform(paperDF['combined_text'].astype(str).tolist())

                all_features = tfidf_vectorizer.get_feature_names_out()
                print('Feature names: %s' % all_features, flush=True)
                print('Number of features: %d' % len(all_features))
                
                # Select top 'k' of the vectorized features.
                selector = SelectKBest(f_classif, k=min(vector_size, x_train.shape[1]))
                selector.fit(x_train, paperDF['binary_decision'])

                print('Selector feature names: %s' % selector.get_feature_names_out(all_features))
            else:
                # Cannot reload with a selector
                assert False
                tfidf_vectorizer = joblib.load(model_file)
                print('Loaded TFIDF vectorizer from: %s' % model_file)
                print('Feature names: %s' % tfidf_vectorizer.get_feature_names_out(), flush=True)
                
                
            metadata_sections, other_sections = peer_useful.divide_sections(sections)
            balancedDF['combined_text'] = balancedDF.apply(lambda x: peer_useful.create_combined_text(x, metadata_sections, other_sections), axis=1)
            balancedDF['combined_text'] = balancedDF['combined_text'].apply(lambda x: x.replace('\n', ' '))

            ## Extract tfidf features for the balanced dataset
            print(balancedDF.head(3)['combined_text'].tolist())
            X_text = lil_matrix(tfidf_vectorizer.transform(balancedDF['combined_text'].astype(str).tolist())).toarray()
            print('Extracted TF*IDF for all balancedDFs: %d' % len(X_text))
            print(X_text[:5])
            
    # Determine the combined values
    if len(sections) > 0:
        if len(numeric_info) > 0:
            # Combine text and numeric features
            X_combined = np.hstack([X_text, X_numeric_scaled])
        else:
            X_combined = X_text
    else:
        X_combined = X_numeric_scaled

    print('Size of X_combined: %s (%s)' % (str(len(X_combined)), type(X_combined)))
    #print('X_combined[0:5]: %s' % X_combined[0:5])
    assert len(X_combined) == len(y)
    print('Checked lengths of X_combined and y', flush=True)
    
    ## Set up Stratified K Fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    # Compute accuracies over splits
    accs = []
    for i, (train_idx, test_idx) in enumerate(skf.split(X_combined, y)):
        print('Creating fold: %d' % i, flush=True)
        X_train = [X_combined[i] for i in train_idx.tolist()]
        y_train = [y[i] for i in train_idx.tolist()]
        print('y_train (%d): %s' % (len(y_train), y_train[:20]))
        X_test = [X_combined[i] for i in test_idx.tolist()]
        y_test = [y[i] for i in test_idx.tolist()]
        if ml_model == 'SVC':
            # Kernel types: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
            clf = SVC(kernel=kernel, random_state=random_state)
        elif ml_model == 'DT':
            clf = DecisionTreeClassifier(max_depth=5, random_state=random_state)
        elif ml_model == 'RF':
            clf = RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1, random_state=random_state
            )
        elif ml_model == 'MLP':
            clf = MLPClassifier(alpha=1, max_iter=1000, random_state=random_state)
        elif ml_model == 'AdaBoost':
            clf = AdaBoostClassifier(random_state=random_state)
        else:
            assert False
    
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        #print('Fold %d (%d): %s' % (i, len(y_test), accuracy_score(y_test, y_pred)))
        #print('y_pred: %s (%s)' % (y_pred[:20], y_test[:20]))
        accs.append(accuracy_score(y_test, y_pred))

    mean_acc = statistics.mean(accs)
    stdev = statistics.stdev(accs)
    print('Accuracy stats: %.2f (%.2f)' % (mean_acc, stdev))
    
    if not os.path.exists(output_file):
        fp = open(output_file, 'w')
        writer = csv.writer(fp)
        writer.writerow(['model', 'vector', 'input_file', 'numeric_info', 'sections', 'vector_size', 'num_folds', 'random_state', 'min_count', 'window', 'epochs', 'kernel', 'lemmatizer', 'min_df', 'max_df', 'ngram_max', 'mean', 'stdev'])
    else:
        fp = open(output_file, 'a')
        writer = csv.writer(fp)
        
    writer.writerow([ml_model, algorithm, args.merged_data.replace('.csv.gz', ''), numeric_info, sections, vector_size, num_folds, random_state, min_count, window, epochs, kernel, lemmatizer, min_df, max_df, ngram_max, mean_acc, stdev])
        
    fp.close()
    print('Produced CSV output to: %s' % output_file)

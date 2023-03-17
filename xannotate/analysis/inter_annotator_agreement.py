import itertools
import math
from itertools import chain
from typing import List, Dict, Hashable, Optional, Union, Any
from typing import Tuple

import numpy as np
import pandas as pd
import glob2 as glob
from sklearn.metrics import cohen_kappa_score, f1_score
from statsmodels.stats.inter_rater import fleiss_kappa

DocMetric = Dict[int, float]
LabelMetric = Dict[str, float]
DocMetricLabel = Dict[str, Dict[str, float]]
DocsAllMetrics = Dict[str, Dict[str, float]]
DocInfo = Dict[Optional[Hashable], Union[Dict[Any, np.ndarray], Dict]]
CountInfo = Dict[Optional[Hashable], Dict]


class InterAnnotatorAgreement:

    def __init__(self, data_directory: str, prefixes: List[str], data_format: str):
        """
        Class that computes inter-annotator agreement metrics

        Args:
            data_directory: e.g. './data'
            data_format: e.g. 'doccano' or 'standard'
            prefixes: [str], e.g. ['test_1B', 'test_1X']
        """
        self.FAILURE_VALUE = -1  # if nan-value is encountered, set the value to FAILURE_VALUE instead
        self.data_directory = data_directory
        self.data_format = data_format
        self.prefixes = prefixes
        self.df, self.doc_ids, self.tags = self._process_data()
        self.doc_dict, self.count_dict = self._get_annotations()
        self.metrics, self.fleiss, self.per_agr, self.cohen, self.label_f1, self.micro_f1 = self._get_metrics()

    def __repr__(self):
        return f"InterAnnotatorAgreement(\nCohen's kappa={self.cohen}\n\nFleiss kappa={self.fleiss}\n\n" \
               f"Average agreement={self.per_agr}\n\nLabel F1={self.label_f1}\n\nMicro_F1={self.micro_f1}\n)"

    def _process_data(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Reads jsonl files from disk and constructs a Panda's dataframe

        Returns:
            df: pd.DataFrame containing information columns with doc_id, char_start, char_end, token, tag
            doc_ids: list of doc ids, e.g. ['24128583', '24088932',...]
            tags: list of tags, e.g. ['SKILL_JOB', 'EXPERIENCE_DURATION']
        """
        dfs = []
        file_paths = [file for prefix in self.prefixes for file in glob.glob(f'{self.data_directory}/*{prefix}*.jsonl')]

        for filepath in file_paths:
            df = pd.read_json(filepath, lines=True)  # converts json file to pandas dataframe
            if self.data_format == 'standard':
                empty_cols = df[df['doc_labels'].str.len() == 0][['doc_id', 'annotator_id']]
                df = pd.DataFrame(dict(doc_id=df.doc_id.values.repeat(df['doc_labels'].str.len()),
                                       annotator_id=df.annotator_id.values.repeat(df['doc_labels'].str.len()))) \
                    .join(pd.DataFrame(df['doc_labels'].sum()))
                df = df.append(empty_cols, ignore_index=True)
                df['doc_id'] = df['doc_id'].astype(str)  # convert doc_ids to strings
            else:  # doccano format
                empty_cols = df[df['labels'].str.len() == 0][['meta', 'annotator_id']]
                df = pd.DataFrame(dict(meta=df.meta.values.repeat(df['labels'].str.len()),
                                       annotator_id=df.annotator_id.values.repeat(df['labels'].str.len()))).join(
                    pd.DataFrame(df['labels'].sum(), columns=['char_start', 'char_end', 'tag']))  # un-nest dataframe
                df = df.append(empty_cols, ignore_index=True)
                df = pd.concat([df.drop(['meta'], axis=1), df['meta'].apply(pd.Series)], axis=1).drop('user_ids',
                                                                                                      axis=1)

            dfs.append(df)
        df = pd.concat(dfs)  # concat all dataframes
        df['annotator_id'] = df['annotator_id'].astype(str)
        doc_ids = list(df.doc_id.unique())  # get the ids of all docs that have been annotated
        tags = list(df.tag.dropna().unique())  # get the unique tags used during the annotation process

        if 'O' not in tags:
            tags.append('O')

        return df, doc_ids, tags

    def _get_annotations(self) -> Tuple[DocInfo, CountInfo]:
        """
        Creates dictionaries with annotations/labels and count of labels (for each job doc)
        in the collected data.

        Returns:
            doc_dct: Dict with keys = doc_ids and values = annotation matrices.
                     An annotation matrix contains the annotations made by each annotator/model.
            count_dct: Dict with keys = doc_ids and values = counts of each tag (e.g. {'SKILL_JOB': 5.0,..}).
        """
        doc_dct = {}
        count_dct = {}
        for frame, data in self.df.groupby('doc_id'):  # frame = meeting id, data = all data for that doc
            annotator_ids = list(data.annotator_id.unique())
            if len(annotator_ids) == 1:
                self.doc_ids.remove(frame)
                continue

            if data['tag'].isnull().all():
                doc_dct[frame] = {}.fromkeys(annotator_ids, np.array([]))
                count_dct[frame] = {}
                continue

            empty_check = (data[data[['char_start', 'char_end']].isna().all(axis=1)])
            # structure data so that the tags made by each annotator is in a separate column
            doc_data = (data.pivot_table(index=['char_start', 'char_end'], columns='annotator_id', values='tag',
                                         aggfunc='first').reset_index())

            if not empty_check.empty:  # handles case when no annotations is made by one of the annotators
                empty_id = empty_check.annotator_id.item()
                doc_data[empty_id] = 'O'

            doc_data = doc_data.fillna('O')  # if tag = NaN replace with label 'O'
            # remove rows where all tags = 'O'
            doc_data = doc_data[~doc_data.filter(items=annotator_ids).apply(set, axis=1).eq({'O'})]
            # count the number of occurrences of each tag
            counts = doc_data.loc[:, doc_data.columns.str.startswith(tuple(annotator_ids))].apply(pd.Series.value_counts) \
                .fillna(0)
            annotations = {k: np.array(doc_data[k]) for k in annotator_ids}  # get the annotations from each annotator
            tag_counts = dict(zip(counts.index, counts.sum(axis=1)))  # tuples containing the count for each tag
            doc_dct[frame] = annotations
            count_dct[frame] = tag_counts

        return doc_dct, count_dct

    def _get_metrics(self) -> Tuple[DocsAllMetrics, DocMetric, DocMetric, DocMetric, DocMetricLabel,
                                    DocMetric]:
        """
        Calculates all inter-annotator metrics such as F1-score, Fleiss' kappa and Cohen's kappa

        Returns:
            metrics: a dictionary containing all metrics for each doc,
                     e.g. {24098116: {cohen: 0.09, fleiss: 0.08,..},..}
                     fleiss, per_agr, cohen, label_f1, micro_f1: dictionaries for each metric,
                     e.g. {24098116: 0.09, 24098117: 0.08}
        """
        metrics, fleiss, label_f1, micro_f1, per_agr, cohen = ({} for i in range(6))

        # Pair-wise IAA metrics ########
        for i in sorted(self.doc_ids):
            annotations = self.doc_dict.get(i)
            vals = np.array(list(annotations.values()))
            fleiss[i] = self.calculate_fleiss(vals)
            per_agr[i] = self.percentage_agreement(vals)
            cohen[i] = self.calculate_cohen_kappa(vals)
            label_f1[i] = self.calculate_label_f1_score(annotations)
            micro_f1[i] = self.calculate_micro_f1_score(annotations)
            metrics[i] = {"fleiss": fleiss[i], "cohen": cohen[i],
                          "per_agr": per_agr[i], "micro_f1": micro_f1[i],
                          "label_f1": label_f1[i]}

        return metrics, fleiss, per_agr, cohen, label_f1, micro_f1

    def calculate_cohen_kappa(self, annotations: np.array) -> float:
        """
        Computes the average pair-wise Cohen's kappa value for one doc

        Args:
            annotations: annotation matrix containing the annotations made by each annotator
                         (for each doc).
                         [size - M x N where M = #annotators and N = #tokens]

        Returns:
            average Cohen's kappa
        """
        if not annotations.size:
            return self.FAILURE_VALUE

        cohen_scores = []
        for pair in itertools.combinations(annotations, 2):  # takes the pair-wise combinations of annotations
            score = cohen_kappa_score(pair[0], pair[1])

            if math.isnan(score):
                print("Encountered NaN value in Cohen's kappa calculation!")
                continue  # skip nan-values for annotators > 2
            cohen_scores.append(score)

        # returns the average Cohen' kappa score across all pair-wise combinations
        avg_cohen = sum(cohen_scores) / len(cohen_scores) if len(cohen_scores) else self.FAILURE_VALUE
        return avg_cohen

    def percentage_agreement(self, annotations: np.array) -> float:
        """
        Computes the average pair-wise Percentage agreement value for one doc
        (number of equal annotations/number of total annotations between two annotators)

        Args:
            annotations: annotation matrix containing the annotations made by each annotator
            (for each doc).
            [size - M x N where M = #annotators and N = #tokens]

        Returns:
            average percentage agreement
        """
        if not annotations.size:
            return self.FAILURE_VALUE

        num_tokens = len(annotations[0])  # number of tokens

        # returns a list with the number of equal annotations for each pair-wise combination of annotations
        agreement = list(chain.from_iterable(
            ((annotations[i + 1:] == row).sum(1)) for i, row in enumerate(annotations[:-1])
        ))

        # average agreement across all pair-wise combinations
        avg_agr = sum((np.array(agreement) / num_tokens)) / len(agreement) if len(agreement) else self.FAILURE_VALUE

        return avg_agr

    def calculate_micro_f1_score(self, annotations: Dict[str, np.array]) -> float:
        """
        Computes the average micro f1-score for one doc

        Args:
            annotations: e.g. {'X': AnnotationMatrix, 'jobtech': AnnotationMatrix'}
                         where AnnotationMatrix = [size - M x N where M = #annotators and N = #tokens]

        Returns:
            the average micro f1-score, e.g. 0.675
        """
        if not np.array(list(annotations.values())).size:
            return self.FAILURE_VALUE

        tags = [x for x in self.tags if x != 'O']  # remove label '0' from evaluation

        # returns a list with the micro f1-score for each pair-wise combination of annotations
        if 'X' in annotations.keys():
            gold_standard = annotations['X']
            p_key = (set(annotations.keys()) - set('X'))
            assert len(p_key) == 1, "ERROR! length of set should be one"
            pred = annotations[p_key.pop()]
            micro_f1_scores = f1_score(gold_standard, pred, labels=tags, average='micro')
        else:
            micro_f1_scores = [f1_score(pair[0], pair[1], labels=tags, average='micro') for pair in
                               itertools.combinations(annotations.values(), 2)]

        # returns the average micro f1-score across all pair-wise combinations
        avg_micro_f1_score = np.mean(micro_f1_scores)

        return avg_micro_f1_score

    def calculate_label_f1_score(self, annotations: Dict[str, np.array]) -> LabelMetric:
        """
        Get the f1-score for each tag for one doc

        Args:
            annotations: annotations: e.g. {'X': AnnotationMatrix, 'jobtech': AnnotationMatrix'}
                         where AnnotationMatrix = [size - M x N where M = #annotators and N = #tokens]

        Returns:
            dictionary with keys = tags, values = average f1-score. E.g. {'SKL': 0.6, EXP: 0.8}
        """
        tags = [x for x in self.tags if x != 'O']  # remove label 'O' from evaluation

        if not np.array(list(annotations.values())).size:
            return {k: self.FAILURE_VALUE for k in tags}

        if 'X' in annotations.keys():  # if gold standard exists
            gold_standard = annotations['X']
            p_key = (set(annotations.keys()) - set('X'))
            assert len(p_key) == 1, "ERROR! length of set should be one"
            pred = annotations[p_key.pop()]
            label_f1_scores_dct = self.model_f1(gold_standard, pred, tags)
        else:
            annotations = np.array(list(annotations.values()))
            label_f1_scores_dct = self.annotator_f1(annotations, tags)

        return label_f1_scores_dct

    def calculate_fleiss(self, annotations: np.array) -> float:
        """
        Computes the Fleiss' kappa value for one doc

        Args:
            annotations: annotation matrix containing the annotations made by each annotator
            (for each doc).
            [size - M x N where M = #annotators and N = #tokens]

        Returns:
            fleiss_dct: the fleiss' kappa value
        """
        if not annotations.size:
            return self.FAILURE_VALUE

        m = self.build_annotation_matrix(annotations)
        self.check_matrix(m, len(annotations))  # assert that matrix is correct

        fleiss_score = fleiss_kappa(m)

        if math.isnan(fleiss_score):
            print('Encountered NaN value in Fleiss calculation!')
            fleiss_score = self.FAILURE_VALUE

        return fleiss_score

    def build_annotation_matrix(self, annotations: np.array) -> np.array:
        """
        Builds the matrix needed to calculate Fleiss' kappa

        Args:
            annotations: annotation matrix containing the annotations made by each annotator
                         (for each doc).
                         [size - M x N where M = #annotators and N = #tokens]

        Returns:
            matrix containing the number of annotations for each token per tag
            [size - N x k where N = #tokens and k = #tags]
        """
        m = []
        for tag in self.tags:
            # counts the number of annotations of one tag per token
            count = np.count_nonzero(annotations == tag, axis=0)
            m.append(count)

        return np.array(m).T

    def model_f1(self, true: np.array, pred: np.array, tags: List[str]) -> LabelMetric:
        """
        Computes the f1-score for each tag between a gold standard
        and model predictions

        Args:
            true: np.array (gold standard labels)
            pred: np.array (label predictions)
            tags: e.g. ['SKILL_JOB', 'EDUCATION_DEGREE'...]

        Returns:
            dictionary with keys = tags, values = average f1-score. E.g. {'SKILL_HARD': 0.6, SKILL_JOB: 0.8}
        """
        f1_scores = []  # stores the f1-scores per label for one pair
        for tag in tags:
            if tag in np.asarray(np.concatenate((true, pred))):
                f1 = f1_score(true, pred, labels=[tag], average=None)
            else:
                f1 = [self.FAILURE_VALUE]
            f1_scores.append(f1[0])

        scores_dct = dict(zip(tags, f1_scores))

        return scores_dct

    def annotator_f1(self, annotations: np.array, tags: List[str]) -> LabelMetric:
        """
        Computes the average pair-wise f1-score for a set of annotators. The f1-score is computed
        for each tag.

        Args:
            annotations: annotations: annotation matrix containing the annotations made by each annotator
                         (for each doc).
                         [size - M x N where M = #annotators and N = #tokens]
            tags: e.g. ['SKILL_JOB', 'SKILL_HARD'..]

        Returns:
            dictionary with keys = tags, values = average f1-score. E.g. {'SKILL_HARD': 0.6, SKILL_JOB: 0.8}
        """
        label_f1_scores = []
        for pair in itertools.combinations(annotations, 2):  # takes the pair-wise combinations of annotations
            f1_scores = []  # stores the f1-scores per label for one pair
            for tag in tags:
                if tag in np.asarray(pair):
                    f1 = f1_score(pair[0], pair[1], labels=[tag], average=None)
                else:
                    f1 = [np.nan]
                f1_scores.append(f1[0])
            label_f1_scores.append(f1_scores)

        label_f1_scores = np.asarray(label_f1_scores).T  # convert list of lists to matrix
        avg_label_f1_scores = [np.nanmean(row) if not np.isnan(row).all() else self.FAILURE_VALUE for row in
                               label_f1_scores]  # returns a list with the average f1-score per tag

        label_f1_scores_dct = dict(zip(tags, avg_label_f1_scores))
        return label_f1_scores_dct

    @staticmethod
    def check_matrix(matrix: np.array, num_annotators: int):
        """
        Assert that the matrix for Fleiss' kappa is correct

        Args:
            matrix: matrix containing the number of annotations for each token per tag
                    [size - N x k where N = #tokens and k = #tags]
            num_annotators: number of annotators
        """
        assert all(
            sum(row) == num_annotators for row in matrix), "Sum of annotations per token are not equal to #annotators!"

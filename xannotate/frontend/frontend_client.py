import json
import os
import shutil

import numpy as np
import pandas as pd
from os.path import join, isfile

from requests.exceptions import ConnectionError
from doccano_client import DoccanoClient
from doccano_client.models.data_upload import Task as DoccanoTask
from typing import List, Dict, Union, Optional, Any, Tuple

from xannotate.frontend.shards.shard_base import ShardBase
from xannotate.frontend.shards.shard import Shard
from xannotate.frontend.shards.shard_user import ShardUser
from xannotate.frontend.config import Config
from xannotate.frontend.project import Project
from xannotate.frontend.batch_id_tracker import BatchIdTracker
from xannotate.analysis import InterAnnotatorAgreement
from xannotate.utils import Utils

import warnings
warnings.filterwarnings("ignore")  # in order to ignore https warnings

BATCH_STAGE_PREVIOUS_MAPPING = {"B": "A", "X": "B"}


class FrontendClient:

    def __init__(self, config_file: str, project_file: str, verbose: bool = False):

        self.config = Config(config_file)
        self.project = Project(project_file)
        self.folder = self.project.file.strip(".jsonl")
        self.tracker = None
        self.dataset, self.batch_nr = self._extract_dataset_and_batch_nr(self.project.file)

        self.NR_ORACLES = len(self.project.annotators)
        self.VERBOSE = verbose
        if verbose:
            print('=== NERBOOTSTRAP FRONTEND CLIENT START ===')
            print(f"> NR_ORACLES = {self.NR_ORACLES} | VERBOSE = {self.VERBOSE}")

        try:
            self.client = DoccanoClient(self.config.url)
            self.client.login(username=self.config.username, password=self.config.password)
            if verbose:
                print(f"> Successfully connected to Doccano server at {self.config.url}")
        except ConnectionError:
            raise Exception(f'! Could not connect to Doccano server at {self.config.url}')

        self.datasets = list()
        self.datasets_overview = list()

        # Utils
        self.utils = Utils()

    @staticmethod
    def _extract_dataset_and_batch_nr(_file_path: str) -> Tuple[str, str]:
        file_name = _file_path.split("/")[-1].strip(".jsonl")   # e.g. test or test_1
        if file_name.count("_") == 1:
            try:
                batch_nr = int(file_name.split("_")[-1])
                dataset = file_name.split("_")[0]
            except Exception as e:
                dataset = file_name
                batch_nr = "1"
        else:
            dataset = file_name
            batch_nr = "1"
        return dataset, batch_nr

    def _start_tracker(self):
        if self.tracker is None:
            os.makedirs(self.folder, exist_ok=True)
            self.tracker = BatchIdTracker(self.folder)

    def _read_guidelines(self) -> str:
        """
        read guidelines and return it as string

        Returns:
            guidelines [str]
        """
        if self.project.guidelines is None:
            guidelines = "no guidelines specified"
        elif isfile(self.project.guidelines):
            with open(self.project.guidelines, "r") as file:
                guidelines = file.read()
        else:
            guidelines = "could not find annotation guidelines"
        return guidelines

    def _read_labels(self) -> Dict[str, str]:
        """
        read labels and return them as dictionary

        Returns:
            labels: e.g. {"SKILL_HARD": "blue", "SKILL_SOFT": "blue_light"}
        """
        if self.project.labels is None:
            labels = "no labels specified"
        elif isfile(self.project.labels):
            with open(self.project.labels, "r") as file:
                labels = json.load(file)
        else:
            labels = "could not find labels"
        return labels

    def overview(self,
                 as_df: bool = True) -> None:
        """
        print an overview of datasets, #documents, #documents_done saved in doccano

        Args:
            as_df:    [bool] e.g. True, return overview as df, otherwise [list] of [dict]
        """
        self._get_datasets_overview(stats=True)
        self._show_datasets_overview(as_df=as_df)

    def first_step(self):
        """
        copy and rename input file
        """
        self._start_tracker()

        new_batch_id = Utils().get_batch_id(self.dataset, self.batch_nr, batch_stage="A")
        renamed_file = join(self.folder, f"{self.dataset}_{self.batch_nr}A_{new_batch_id}.jsonl")

        self.tracker.dump(batch_stage="A", batch_id=new_batch_id)
        shutil.copy2(self.project.file, renamed_file)

    def last_step(self):
        """
        copy and rename output file
        """
        self._start_tracker()
        batch_id = self.tracker.get_batch_id(batch_stage="X")
        output_file = join(self.folder, f"{self.dataset}_{self.batch_nr}X_{batch_id}.jsonl")
        renamed_output_file = self.project.file.replace(".jsonl", "_ANNOTATED.jsonl")

        shutil.copy2(output_file, renamed_output_file)

    ####################################################################################################################
    # CONVERT LOCAL
    ####################################################################################################################
    def convert2doccano(self,
                        batch_stage: str) -> None:
        """
        convert locally from standard to doccano format

        Args:
            batch_stage: [str], e.g. 'B'
        """
        self._start_tracker()

        if batch_stage == "A":
            shard = Shard(
                folder=self.folder,
                dataset=self.dataset,
                batch_id=self.tracker.get_batch_id(batch_stage),
                batch_nr=self.batch_nr,
                batch_stage="A",
            )
            self._convert2doccano(shard)
        else:
            raise Exception(f"ERROR! batch_stage = {batch_stage} not allowed.")

    def _convert2doccano(self,
                         shard: Shard):

        def _convert_labels_2_doccano(labels: List[Dict[str, str]]) -> List[List[str]]:
            return [
                [int(label["char_start"]), int(label["char_end"]), label["tag"]]
                for label in labels
            ]

        # load batch from local
        docs_model_annotated = self._load_documents_from_local(local_path=shard.get_path_standard())

        # convert
        docs_doccano = list()

        for doc in docs_model_annotated:
            doc_doccano = {
                "text": doc["text"],
                "labels": _convert_labels_2_doccano(doc["tags"]),
                # metadata:
                "doc_id": doc["doc_id"],
                "class_system": "",
                "batch_history": [],
            }
            docs_doccano.append(doc_doccano)

        # save
        self._save_documents_to_local(documents=docs_doccano,
                                      local_path=shard.get_path_doccano(part="doccano"))

    def convert2standard(self,
                         batch_stage: Optional[str] = None) -> None:
        """
        convert locally from doccano to standard format

        Args:
            batch_stage: [str], e.g. 'B'
        """
        self._start_tracker()

        if batch_stage == "B":
            for user_id in range(1, self.NR_ORACLES + 1):
                shard_user = ShardUser(
                    folder=self.folder,
                    dataset=self.dataset,
                    batch_id=self.tracker.get_batch_id(batch_stage),
                    batch_nr=self.batch_nr,
                    batch_stage="B",
                    user_id=str(user_id),
                )
                self._convert2standard(shard_user, user_id=user_id)
        elif batch_stage == "X":
            shard = Shard(
                folder=self.folder,
                dataset=self.dataset,
                batch_id=self.tracker.get_batch_id(batch_stage),
                batch_nr=self.batch_nr,
                batch_stage="X",
            )
            self._convert2standard(shard)
        else:
            raise Exception(f"ERROR! batch_stage = {batch_stage} not allowed.")

    def _convert2standard(self,
                          shard: Union[Shard, ShardUser],
                          user_id: Optional[int] = None):

        def _convert_labels_2_standard(labels: List[List[str]], text: str) -> List[Dict[str, str]]:
            return [
                {
                    "char_start": str(label[0]),
                    "char_end": str(label[1]),
                    "tag": label[2],
                    "token": text[label[0]: label[1]],
                }
                for label in labels
            ]

        # load batch from local
        docs_user_annotated = self._load_documents_from_local(
            local_path=shard.get_path_doccano(part="doccano"))
        docs_metadata = self._load_documents_from_local(local_path=shard.get_path_doccano(part="metadata"))[0]

        if user_id is not None:
            assert isinstance(shard, ShardUser), \
                f"ERROR! user_id = {user_id} is not None, but shard is not ShardUser"

            # change user_id
            def _value(_k, _v, _user_id):
                return f"{_user_id}" if _k == "user_ids" else _v

            docs_metadata["batch_history"] = [
                {k: _value(k, v, user_id) for k, v in event.items()}
                for event in docs_metadata["batch_history"]
            ]

        # convert
        docs_standard = list()
        for doc in docs_user_annotated:
            doc_standard = {k: v for k, v in docs_metadata.items()}
            doc_standard["doc_id"] = doc["meta"]["doc_id"]
            doc_standard["doc_text"] = doc["text"]
            doc_standard["doc_labels"] = _convert_labels_2_standard(doc["labels"], doc["text"])
            doc_standard["annotator_id"] = doc["annotator_id"]
            doc_standard["class_system"] = doc["meta"]["class_system"]
            docs_standard.append(doc_standard)

        # save
        self._save_documents_to_local(documents=docs_standard,
                                      local_path=shard.get_path_standard())

    ####################################################################################################################
    # SPLIT
    ####################################################################################################################
    def split(self):
        """
        split doccano batch data into shards
        """
        self._start_tracker()

        # load
        shard = Shard(
            folder=self.folder,
            dataset=self.dataset,
            batch_id=self.tracker.get_batch_id("A"),
            batch_nr=self.batch_nr,
            batch_stage="A",
        )
        docs_model_annotated = self._load_documents_from_local(local_path=shard.get_path_doccano(part="doccano"))

        # split
        parts = [list(elem) for elem in np.array_split(docs_model_annotated, self.NR_ORACLES)]
        docs_model_annotated_split = {
            user_id: parts[user_id-1 % self.NR_ORACLES] + parts[user_id % self.NR_ORACLES]
            for user_id in range(1, self.NR_ORACLES+1)
        }

        # save
        for user_id in range(1, self.NR_ORACLES+1):
            shard_user = ShardUser(
                folder=self.folder,
                dataset=self.dataset,
                batch_id=self.tracker.get_batch_id("A"),
                batch_nr=self.batch_nr,
                batch_stage="A",
                user_id=str(user_id),
            )
            self._save_documents_to_local(documents=docs_model_annotated_split[user_id],
                                          local_path=shard_user.get_path_doccano(part="doccano"))

    ####################################################################################################################
    # MERGE
    ####################################################################################################################
    def local_merge(self):
        """
        merge different annotations
        """
        self._start_tracker()

        batch_id_a = self.tracker.get_batch_id("A")
        batch_id_b = self.tracker.get_batch_id("B")

        # compute IAA
        iaa = InterAnnotatorAgreement(
            data_directory=self.folder,
            prefixes=[f"{self.dataset}_{self.batch_nr}B_{batch_id_b}_user"],
            data_format="doccano",
        )

        # load batch_stage = A model annotations
        shard = Shard(
            folder=self.folder,
            dataset=self.dataset,
            batch_id=batch_id_a,
            batch_nr=self.batch_nr,
            batch_stage="A",
        )
        docs_model_annotated = self._load_documents_from_local(local_path=shard.get_path_doccano(part="doccano"))

        # usually, metadata is included at top level of the dict
        # however, here, we temporarily move it to a 'meta' dict (will be reverted below)
        for doc_model_annotated in docs_model_annotated:
            meta = {
                k: v
                for k, v in doc_model_annotated.items()
                if k not in ["text", "labels"]
            }
            for k in meta.keys():
                doc_model_annotated.pop(k)
            doc_model_annotated["meta"] = meta

        # load batch_stage = B single user annotations
        docs_user_annotated_single = dict()
        for user_id in range(1, self.NR_ORACLES+1):
            shard_user = ShardUser(
                folder=self.folder,
                dataset=self.dataset,
                batch_id=batch_id_b,
                batch_nr=self.batch_nr,
                batch_stage="B",
                user_id=str(user_id),
            )
            docs_user_annotated_single[user_id] = \
                self._load_documents_from_local(local_path=shard_user.get_path_doccano(part="doccano"))

        # build batch_stage = B merged single user annotations
        docs_user_annotated = list()
        for doc_model in docs_model_annotated:
            doc_id = doc_model["meta"]["doc_id"]
            doc_class_system = doc_model["meta"]["class_system"]

            doc_user_annotated = {
                "text": doc_model["text"],
                "labels": None,
                "meta": {
                    "doc_id": doc_id,
                    "user_ids": [],
                    "class_system": doc_class_system,
                }
            }
            all_labels = list()
            for user_id in range(1, self.NR_ORACLES+1):
                for doc_user in docs_user_annotated_single[user_id]:
                    if doc_user["meta"]["doc_id"] == doc_model["meta"]["doc_id"]:
                        doc_user_annotated["meta"]["user_ids"].append(user_id)
                        all_labels.extend(doc_user["labels"])
            doc_user_annotated["meta"]["user_ids"] = \
                ",".join([str(elem) for elem in doc_user_annotated["meta"]["user_ids"]])

            def unify_labels(_all_labels: list):
                unified_labels = list()
                for label in _all_labels:
                    count = _all_labels.count(label)
                    if count == 1:
                        label[-1] = "???"
                        if label not in unified_labels:
                            unified_labels.append(label)
                    elif count == 2 and label not in unified_labels:
                        unified_labels.append(label)
                return unified_labels

            doc_user_annotated["labels"] = unify_labels(all_labels)
            doc_user_annotated["meta"]["IAA (micro f1)"] = float(np.round(iaa.micro_f1[doc_id], 3))
            docs_user_annotated.append(doc_user_annotated)

        # move meta data to top level of dict (revert operation from above)
        for doc_user_annotated in docs_user_annotated:
            meta = doc_user_annotated["meta"]
            for k, v in meta.items():
                doc_user_annotated[k] = v
            doc_user_annotated.pop("meta")

        # save batch_stage = B merged single user annotations
        shard = Shard(
            folder=self.folder,
            dataset=self.dataset,
            batch_id=batch_id_b,
            batch_nr=self.batch_nr,
            batch_stage="B",
        )
        self._save_documents_to_local(documents=docs_user_annotated,
                                      local_path=shard.get_path_doccano(part="doccano"))

    ####################################################################################################################
    # LOCAL -> DOCCANO
    ####################################################################################################################
    def doccano_post(self,
                     batch_stage: str,
                     ):
        """
        post dataset batch to doccano
        local input file is DATA_DIR_DOCCANO/DATASET_ID_batch_BATCH_NR_input.jsonl_doccano

        Args:
            batch_stage: [str], e.g. 'B'
        """
        self._start_tracker()

        if batch_stage == "A":
            for user_id in range(1, self.NR_ORACLES+1):
                shard = ShardUser(
                    folder=self.folder,
                    dataset=self.dataset,
                    batch_id=self.tracker.get_batch_id(batch_stage),
                    batch_nr=self.batch_nr,
                    batch_stage=batch_stage,
                    user_id=str(user_id),
                )
                self._doccano_post(shard, assign=True)

        elif batch_stage == "B":
            shard = Shard(
                folder=self.folder,
                dataset=self.dataset,
                batch_id=self.tracker.get_batch_id(batch_stage),
                batch_nr=self.batch_nr,
                batch_stage=batch_stage,
            )
            self._doccano_post(shard)
        else:
            raise Exception(f"ERROR! post is only defined for batch_stage = A, B, not {batch_stage}")

    def _doccano_post(self,
                      shard: ShardBase,
                      assign: bool = False):

        # 0. make sure project does not yet exist
        self._get_datasets_overview(stats=False)
        assert shard.name not in self.datasets, f'dataset {shard.name} already exists => doccano_post failed'

        # 1. make sure necessary files exist
        input_file_documents = shard.get_path_doccano(part="doccano")
        assert isfile(input_file_documents), f'input file {input_file_documents} does not exist => doccano_post failed'

        # 2. read documents from jsonl
        documents = self._load_documents_from_local(local_path=input_file_documents)
        print(f"# documents = ", len(documents))
        print(f"labels:", documents[0]["labels"])
        # print(documents[0]["metadata"])

        # check labelled token boundaries
        self._check_labelled_token_boundaries(documents, "doccano", assert_correctness=False)

        # 3. create project
        msg_create_project = self.client.create_project(
            name=shard.name,
            project_type='SequenceLabeling',
            description="my description",
            guideline=self._read_guidelines(),
            collaborative_annotation=True,
        )
        print("> create project", msg_create_project)
        project_id = self._get_project_id_for_dataset(dataset_id=shard.name)

        # 4. get label info for doccano
        label_specs = Utils.get_label_info_for_doccano(
            labels=self._read_labels(),
        )

        if shard.batch_stage == "B":
            label_specs.append({
                "text": "???",
                "suffix_key": "",
                "background_color": "#f7ff00",  # neon yellow
                "text_color": "#ffffff",
            })

        # 5. import labels
        for label_spec in label_specs:
            msg_create_label = self.client.create_label_type(
                project_id,
                "span",
                text=label_spec['text'],
                color=label_spec['background_color'],
            )
            print("> create label:", msg_create_label)

        # 5. import documents
        self.client.upload(
            project_id,
            [input_file_documents],
            DoccanoTask.SEQUENCE_LABELING,
            "JSONL",
            "text",
            "labels",
        )

        if assign:
            assert isinstance(shard, ShardUser), f"ERROR! shards can only be assigned to single annotator."
            self._doccano_assign(shard)

        self._done(msg=f"> succesfully posted id={project_id} | name={shard.name} from local 2 doccano")

    def _doccano_assign(self,
                        shard: ShardBase):

        project_id = self._get_project_id_for_dataset(dataset_id=shard.name)
        user_id = f"user{shard.user_id}"  # TODO: generalize

        self.client.add_member(project_id, user_id, "annotator")

    ####################################################################################################################
    # LOCAL <- DOCCANO
    ####################################################################################################################
    def doccano_get(self,
                    batch_stage: str,
                    ):
        """
        get dataset batch from doccano
        local output file is DATA_DIR_DOCCANO/DATASET_ID_batch_BATCH_NR_output.jsonl_doccano

        Args:
            batch_stage: [str], e.g. 'B'
        """
        self._start_tracker()

        previous_batch_stage = BATCH_STAGE_PREVIOUS_MAPPING[batch_stage]
        previous_batch_id = self.tracker.get_batch_id(batch_stage=previous_batch_stage)

        new_batch_id = Utils().get_batch_id(self.dataset, self.batch_nr, batch_stage)

        if batch_stage == "B":
            for user_id in range(1, self.NR_ORACLES+1):
                shard = ShardUser(
                    folder=self.folder,
                    dataset=self.dataset,
                    batch_id=previous_batch_id,
                    batch_nr=self.batch_nr,
                    batch_stage=batch_stage,
                    user_id=str(user_id),
                )
                self._doccano_get(shard, new_batch_id=new_batch_id)

        elif batch_stage == "X":
            shard = Shard(
                folder=self.folder,
                dataset=self.dataset,
                batch_id=previous_batch_id,
                batch_nr=self.batch_nr,
                batch_stage=batch_stage,
            )
            self._doccano_get(shard, new_batch_id=new_batch_id)
        else:
            raise Exception(f"ERROR! get is only defined for batch_stage = B, X not {batch_stage}")

        self.tracker.dump(batch_stage=batch_stage, batch_id=new_batch_id)

    def _doccano_get(self,
                     shard: Union[Shard, ShardUser],
                     new_batch_id: Optional[str] = None):
        # -1. set batch_stage to previous
        shard.set_batch_stage(shard.previous_batch_stage())

        # 0. make sure project does exist
        self._get_datasets_overview(stats=False)
        assert shard.name in self.datasets, f'dataset {shard.name} does not exist => export failed'

        # 1. check that all documents are tagged
        project_id = self._get_project_id_for_dataset(dataset_id=shard.name)
        nrdocuments, nrdocumentsdone, nrunresolved = self._get_annotation_status(project_id=project_id)
        assert nrdocuments == nrdocumentsdone, f'only {nrdocumentsdone} of {nrdocuments} tagged => export failed'
        assert nrunresolved == 0, f'there exist {nrunresolved} tags ??? => export failed'

        # 2. get documents & metadata
        project_id = self._get_project_id_for_dataset(dataset_id=shard.name)

        # a. documents
        document_list = self._get_documents(
            project_id=project_id,
        )

        documents = list()
        for document_list_elem in document_list:
            labels = [
                [
                    annotation.start_offset,
                    annotation.end_offset,
                    self.client.find_label_type_by_id(project_id=project_id,
                                                      label_type_id=annotation.label,
                                                      type="span").text
                ]
                for annotation in document_list_elem['annotations']
            ]
            document: Dict[str, Any] = {
                'text': document_list_elem['text'],
                'labels': sorted(labels, key=lambda x: x[0]),
                'meta': document_list_elem['meta'],
            }
            if isinstance(shard, ShardUser):
                document["meta"]["user_ids"] = f"{shard.user_id}"
                document["annotator_id"] = f"{shard.user_id}"
            elif isinstance(shard, Shard):
                document["meta"]["user_ids"] = "X"
                document["annotator_id"] = "X"
                document["meta"].pop("IAA (micro f1)")
            documents.append(document)

        # b. metadata
        # metadata = self._load_documents_from_local(local_path=shard.get_path_doccano(part="metadata"))[0]
        metadata = {"batch_history": list(), "batch_stage": shard.next_batch_stage()}
        if isinstance(shard, ShardUser):
            batch_entry = {
                "date": Utils.date(),
                "type": "user annotation single",
                "user_ids": "ORACLE_ID",
            }
            annotator_id = "ORACLE_ID"
        elif isinstance(shard, Shard):
            batch_entry = {
                "date": Utils.date(),
                "type": "user annotation",
                "user_ids": "X",
            }
            annotator_id = "X"
        else:
            raise Exception(f"ERROR! shard type = {type(shard)} is unexpected.")
        metadata["batch_history"].append(batch_entry)
        metadata["annotator_id"] = annotator_id

        if self.VERBOSE:
            print("=== metadata ===")
            print(metadata)
            print("================")

        # c. doccano
        doccano = None
        if isinstance(shard, ShardUser):
            doccano = self._load_documents_from_local(local_path=shard.get_path_doccano(part="doccano_previous"))

        # 3. save documents & metadata to local

        # new batch_id
        if new_batch_id is not None:
            shard.set_batch_id(new_batch_id)

        # c. doccano
        if isinstance(shard, ShardUser):
            local_path = shard.get_path_doccano(part="doccano_previous")
            self._save_documents_to_local(documents=doccano, local_path=local_path)

        # new batch_stage
        shard.set_batch_stage(shard.next_batch_stage())

        # a. documents
        # check labelled token boundaries
        documents_corrected = self._check_labelled_token_boundaries(documents, "doccano", assert_correctness=False)

        local_path = shard.get_path_doccano(part="doccano")
        self._save_documents_to_local(documents=documents_corrected, local_path=local_path)

        # b. metadata
        metadata["batch_id"] = shard.batch_id
        local_path = shard.get_path_doccano(part="metadata")
        self._save_documents_to_local(documents=[metadata], local_path=local_path)

        self._done(msg=f"> succesfully got id={project_id} | name={shard.name} from doccano 2 local")

    ####################################################################################################################
    # HELPER FUNCTIONS #################################################################################################
    ####################################################################################################################
    def _check_labelled_token_boundaries(self,
                                         documents: List[Dict],
                                         data_format: str,
                                         assert_correctness: bool) -> Union[None, List[Dict]]:
        """
        check that labelled tokens in data do not contain whitespace or punctuation.

        Args:
            documents:             [list] of [AnnotatedAd]
            data_format:           [str] "doccano" or "standard"
            assert_correctness:    [bool] if True, expect that boundaries are correct and raise Exception if not.
                                          if False, allow for adjustments

        Returns:
            documents_corrected: [list] of [AnnotatedAd] if assert_correctness is False
        """
        assert data_format in ["doccano", "standard"], \
            f"ERROR! method only usable w/ data_format = doccano or standard, not {data_format}"

        check, documents_corrected = self.utils.check_labelled_token_boundaries(documents, data_format)

        if assert_correctness is True:
            if check is False:
                raise Exception(f"ERROR! labelled token boundary problem found!")
            return None
        else:
            if check is False:
                print(f"> ATTENTION! labelled token boundaries were adapted!")
            return documents_corrected

    def _get_datasets_overview(self,
                               stats: bool = False) -> None:
        """
        Args:
            stats: if true, get #documents, #documents_done, #???

        Created Attr:
            datasets_overview [list] of [dict] w/ keys 'id', 'name', '#documents', '#documents_done'
            datasets          [list] of [str] dataset_ids / 'name' from datasets_overview
        """
        self.datasets_overview = [
            {'id': elem.id, 'name': elem.name}
            for elem in self.client.list_projects()  # returns [list] of [dict] with projects
        ]
        if stats:
            for i, dataset in enumerate(self.datasets_overview):
                self.datasets_overview[i]['#documents'], \
                    self.datasets_overview[i]['#documents_done'], \
                    self.datasets_overview[i]['#???'] = \
                    self._get_annotation_status(project_id=dataset['id'])

        self.datasets = [elem['name'] for elem in self.datasets_overview]

    def _get_annotation_status(self,
                               project_id: int) -> Tuple[int, int, int]:
        """
        Args:
            project_id:       [int]

        Returns:
            nrdocuments     [int] number of documents in project/dataset
            nrdocumentsdone [int] number of documents in project/dataset that have been tagged
            nrunresolved    [int] number of unresolved tags (???)
        """
        document_list = self._get_documents(
            project_id=project_id,
        )
        label_ids = list()
        for document_list_elem in document_list:
            for annotation in document_list_elem['annotations']:
                label_ids.append(annotation.label)
        label_ids_unique = list(set(label_ids))

        label_id_unresolved = None
        for label_id in label_ids_unique:
            if self.client.find_label_type_by_id(
                    project_id=project_id,
                    label_type_id=label_id,
                    type='span').text == "???":
                label_id_unresolved = label_id

        if label_id_unresolved is None:
            nrunresolved = 0
        else:
            nrunresolved = label_ids.count(label_id_unresolved)

        nrdocuments = len(document_list)
        nrdocumentsdone = \
            sum([1
                 for elem in document_list
                 if elem['is_confirmed'] is True
                 ])
        return nrdocuments, nrdocumentsdone, nrunresolved

    def _get_documents(self,
                       project_id: int) -> List[Dict]:
        """
        Args:
            project_id:     [int]

        Returns:
            document_list [list] of examples
        """
        examples = self.client.list_examples(
            project_id=project_id,
        )
        documents = [vars(example) for example in examples]
        for document in documents:
            document["annotations"] = self.client.list_spans(project_id, document["id"])

        return documents

    def _show_datasets_overview(self,
                                as_df: bool = False) -> None:
        """
        Args:
            as_df: [bool] if True, return pandas dataframe
        """
        print('=== DOCCANO DATASETS OVERVIEW ===')
        if as_df:
            df = pd.DataFrame(self.datasets_overview)
            if len(df):
                print(df)
            else:
                print("no data found")
        else:
            print(self.datasets_overview)

    def _get_project_id_for_dataset(self,
                                    dataset_id: str) -> int:
        """
        Args:
            dataset_id: [str]

        Returns:
            project_id: [int]
        """
        self._get_datasets_overview(stats=False)
        project_id = None
        for dataset in self.datasets_overview:
            if dataset['name'] == dataset_id:
                project_id = dataset['id']
                break

        assert project_id is not None, f'could not find project_id for dataset_id = {dataset_id}'
        return project_id

    @staticmethod
    def _save_documents_to_local(documents: List[Dict], local_path: str) -> None:
        """
        save documents to local hard drive

        Args:
            documents:  e.g. [{'dataset': 'test', ..}, ..]
            local_path: e.g. '.data/test_5B_123456789.jsonl'
        """
        with open(local_path, 'w') as file:
            for document in documents:
                file.write(json.dumps(document, ensure_ascii=False) + "\n")
        print(f'> {len(documents)} documents / entries saved at {local_path}')

    @staticmethod
    def _load_documents_from_local(local_path: str) -> List[Dict]:
        """
        load documents from local hard drive

        Args:
            local_path:  e.g. './data/test_5B_123456789.jsonl'

        Returns:
            documents: e.g. [{'dataset': 'test', ..}, ..]
        """
        with open(local_path, 'r') as file:
            jlines = file.readlines()
            documents = [json.loads(jline) for jline in jlines]
        return documents

    @staticmethod
    def _done(msg: str) -> None:
        print()
        print(msg)
        print('=== DOCCANO END ===')

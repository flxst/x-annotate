from abc import ABC, abstractmethod


class ShardBase(ABC):

    def __init__(self,
                 folder: str,
                 dataset: str,
                 batch_id: str,
                 batch_nr: str,
                 batch_stage: str,
                 **kwargs,
                 ):
        """
        Args:
            folder:      [str], e.g. 'data/test'
            dataset:     [str], e.g. 'swedish_ner_corpus'
            batch_id:    [str], e.g. '12345'
            batch_nr:    [str], e.g. '5'
            batch_stage: [str], e.g. 'B'
        """
        self.folder = folder
        self.dataset = dataset
        self.batch_id = batch_id
        self.batch_nr = batch_nr
        self.batch_stage = batch_stage

        self.labels = self.dataset.split("-")[0]
        self.name_core = None
        self.name = None
        self.dict = None
        self.user_id = kwargs["user_id"] if "user_id" in kwargs.keys() else None
        self._derive_attributes()

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    @abstractmethod
    def _derive_attributes(self):
        pass

    ####################################################################################################################
    # BASE METHODS
    ####################################################################################################################
    def get_path_doccano(self,
                         part: str) -> str:
        """
        Args:
            part:        [str], 'doccano' or 'metadata'

        Returns:
            path:      [str], e.g. 'data/TEST_batch_5_input.jsonl_doccano'
        """
        assert part in ["doccano", "metadata", "doccano_previous"], \
            f"ERROR! part = {part} needs to be doccano or metadata"
        if part == "doccano":
            return f'{self.folder}/{self.name}__{part}.jsonl'
        elif part == "metadata":
            return f'{self.folder}/{self.name_core}__{part}.jsonl'
        elif part == "doccano_previous":
            return f'{self.folder}/{self.name_core}__doccano.jsonl'

    def set_batch_stage(self,
                        batch_stage: str):
        """
        Args:
            batch_stage: [str], e.g. 'B'
        """
        self.batch_stage = batch_stage
        self._derive_attributes()

    def set_batch_id(self,
                     batch_id: str):
        """
        change batch_id using hash
        """
        self.batch_id = batch_id
        self._derive_attributes()

    def next_batch_stage(self):
        if self.batch_stage == "A":
            return "B"
        elif self.batch_stage == "B":
            return "X"
        else:
            raise Exception(f"ERROR! next_batch_stage work only for current batch stage A & B (not {self.batch_stage})")

    def previous_batch_stage(self):
        if self.batch_stage == "X":
            return "B"
        if self.batch_stage == "B":
            return "A"
        else:
            raise Exception(f"ERROR! next_batch_stage work only for current batch stage B & X (not {self.batch_stage})")

    def get_path_standard(self) -> str:
        """
        Returns:
            path:      [str], e.g. 'data/TEST_5A_123456789.jsonl'
        """
        assert self.batch_stage in ["A", "B", "X"], f"ERROR! batch_stage = {self.batch_stage} needs to be A, B or X"
        if self.batch_stage in ["A", "X"]:
            return f'{self.folder}/{self.name_core}.jsonl'
        elif self.batch_stage in ["B"]:
            return f'{self.folder}/{self.name}.jsonl'

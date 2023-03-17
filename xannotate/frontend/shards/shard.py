from xannotate.frontend.shards.shard_base import ShardBase


class Shard(ShardBase):

    def __init__(self,
                 folder: str,
                 dataset: str,
                 batch_id: str,
                 batch_nr: str,
                 batch_stage: str,
                 ):
        """
        Args:
            dataset:         [str], e.g. 'swedish_ner_corpus'
            batch_id:        [str], e.g. '12345'
            batch_nr:        [str], e.g. '5'
            batch_stage:     [str], e.g. 'B'
        """
        super().__init__(folder, dataset, batch_id, batch_nr, batch_stage)

    def _derive_attributes(self):
        self.name_core = f"{self.dataset}_{self.batch_nr}{self.batch_stage}_{self.batch_id}"
        self.name = f"{self.dataset}_{self.batch_nr}{self.batch_stage}_{self.batch_id}"
        self.dict = {
            "dataset": self.dataset,
            "batch_id": self.batch_id,
            "batch_nr": self.batch_nr,
            "batch_stage": self.batch_stage,
        }

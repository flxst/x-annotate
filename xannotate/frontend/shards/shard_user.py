from xannotate.frontend.shards.shard_base import ShardBase


class ShardUser(ShardBase):

    def __init__(self,
                 folder: str,
                 dataset: str,
                 batch_id: str,
                 batch_nr: str,
                 batch_stage: str,
                 user_id: str,
                 ):
        """
        Args:
            user_id: [int], e.g. 1
        """
        super().__init__(folder, dataset, batch_id, batch_nr, batch_stage, user_id=user_id)
        self.user_id = user_id

    def _derive_attributes(self):
        self.name_core = f"{self.dataset}_{self.batch_nr}{self.batch_stage}_{self.batch_id}"
        self.name = f"{self.dataset}_{self.batch_nr}{self.batch_stage}_{self.batch_id}_user{self.user_id}"
        self.dict = {
            "dataset": self.dataset,
            "batch_id": self.batch_id,
            "batch_nr": self.batch_nr,
            "batch_stage": self.batch_stage,
            "user_id": self.user_id,
        }

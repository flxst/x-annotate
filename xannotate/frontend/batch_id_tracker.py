from os.path import join, isfile, isdir
import json
from typing import Dict


class BatchIdTracker:

    def __init__(self, folder: str):
        assert isdir(folder), f"ERROR! folder = {folder} does not exist."
        self.file_path = join(folder, "batch_ids.json")

    def _read(self) -> Dict[str, str]:
        if isfile(self.file_path):
            with open(self.file_path, "r") as f:
                tracker_dict = json.load(f)
        else:
            tracker_dict = dict()
        return tracker_dict

    def _write(self, tracker_dict: Dict[str, str]) -> None:
        with open(self.file_path, "w") as f:
            json.dump(tracker_dict, f)

    def dump(self, batch_stage: str, batch_id: str):
        tracker_dict = self._read()
        assert batch_stage not in tracker_dict.keys(), f"ERROR! there is a file with batch_stage = {batch_stage}"
        tracker_dict[batch_stage] = batch_id
        self._write(tracker_dict)

    def get_batch_id(self, batch_stage: str) -> str:
        tracker_dict = self._read()
        assert batch_stage in tracker_dict.keys(), f"ERROR! there is no file with batch_stage = {batch_stage}"
        return tracker_dict[batch_stage]

import configparser
from os.path import isfile


class Project:

    def __init__(self, project_file: str):
        project = configparser.ConfigParser()
        project.read(project_file)
        self.name = project["data"]["name"]
        self.file = project["data"]["file"]
        self.guidelines = project["metadata"]["guidelines"] if "guidelines" in project["metadata"].keys() else None
        self.labels = project["metadata"]["labels"] if "labels" in project["metadata"].keys() else None
        self.master = project["master"]["username"]
        self.annotators = {i: annotator for i, annotator in enumerate(project["annotators"]["username"].split(","))}

        assert self.file.endswith(".jsonl"), f"ERROR! file path = {self.file} needs to end with .jsonl"
        assert isfile(self.file), f"ERROR! file at {self.file} does not exist"

    def __repr__(self):
        annotators_str = ",".join([annotator for _, annotator in self.annotators.items()])
        return f"name = {self.name} | " \
               f"file = {self.file} | " \
               f"master = {self.master} | " \
               f"annotators = {annotators_str}"

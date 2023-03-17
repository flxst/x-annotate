
from datetime import datetime
import sys
from typing import List, Dict, Any, Union

from xannotate.utils.colors import COLORS

AnnotatedDoc = Dict[str, Any]
LabelDoccano = List[Union[int, str]]
LabelStandard = Dict[str, str]


class Utils:

    def __init__(self):
        pass

    @staticmethod
    def check_labelled_token_boundaries(data: List[AnnotatedDoc],
                                        data_format: str) -> (bool, List[AnnotatedDoc]):
        """
        check that labelled tokens in data do not contain whitespace or line breaks.

        Args:
            data:             [list] of [AnnotatedDoc]
            data_format:      [str] "doccano" or "standard"

        Returns:
            check:          [bool] True if no changes were necessary, False otherwise
            data_corrected: [list] of [AnnotatedDoc]
        """
        line_break_characters = ['\n', '\r']

        data_corrected = list()
        adjusted_labels = list()
        for ad in data:
            ad_labels_corrected = list()
            labels = ad["labels"] if data_format == "doccano" else ad["ad_labels"]
            for label in labels:
                if data_format == "doccano":
                    char_start = int(label[0])
                    char_end = int(label[1])
                    tag = label[2]
                    token = ad["text"][char_start: char_end]
                else:  # standard
                    char_start = int(label["char_start"])
                    char_end = int(label["char_end"])
                    tag = label["tag"]
                    token = label["token"]

                change = False
                while 1:
                    if token.startswith(tuple(" ") + tuple(line_break_characters)):
                        token = token[1:]
                        char_start += 1
                        change = True
                    else:
                        break

                while 1:
                    if token.endswith(tuple(" ") + tuple(line_break_characters)):
                        token = token[:-1]
                        char_end -= 1
                        change = True
                    else:
                        break

                if change:
                    adjusted_labels.append(token)
                    if data_format == "doccano":
                        label = [char_start, char_end, tag]
                    else:
                        label = {
                            "char_start": str(char_start),
                            "char_end": str(char_end),
                            "tag": tag,
                            "token": token,
                        }

                ad_labels_corrected.append(label)

            ad_corrected = ad
            if data_format == "doccano":
                ad_corrected["labels"] = ad_labels_corrected
            else:
                ad_corrected["ad_labels"] = ad_labels_corrected
            data_corrected.append(ad_corrected)

        if len(adjusted_labels):
            print(f"> {len(adjusted_labels)} adjusted labels:")
            for adjusted_label in adjusted_labels:
                print(f"  {adjusted_label}")
            check = False
        else:
            check = True
            print(f"> no adjusted labels")

        return check, data_corrected

    @staticmethod
    def date() -> str:
        return datetime.now().strftime("%Y-%m-%d|%H:%M:%S")

    def get_batch_id(self,
                     dataset: str,
                     batch_nr: str,
                     batch_stage: str) -> str:
        hash_input = f"{dataset}_{batch_nr}{batch_stage}_{datetime.now()}"
        return self.custom_hash(hash_input)

    @staticmethod
    def custom_hash(hash_input):
        return str(hash(hash_input) % ((sys.maxsize + 1) * 2))[:9]

    @staticmethod
    def get_label_info_for_doccano(labels: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        create json file for oracle

        Args:
            labels: e.g. {"SKILL_HARD": "blue", "SKILL_SOFT": "blue_light"}

        Returns:
            label_info: list of dictionaries that contain label information
        """
        def get_color(_value):
            if _value.startswith("#"):
                return _value
            elif _value in COLORS.keys():
                return COLORS[_value]
            else:
                return '#808080'  # 808080 = grey

        label_info = [
            {
                'text': key,
                'suffix_key': key.lower()[0],
                'background_color': get_color(value),
                'text_color': '#ffffff',  # ffffff = white
            }
            for key, value in labels.items()
        ]

        return label_info

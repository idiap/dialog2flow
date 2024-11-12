# -*- coding: utf-8 -*-
"""
Script to convert DialogStudio TOD datasets into a common standardized unified format.

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import re
import json
import nltk
import argparse

from tqdm import tqdm
from nltk.corpus import wordnet
from collections import Counter
from datasets import load_dataset
from huggingface_hub import login
from nltk.stem import WordNetLemmatizer

# e.g. python create_tod_datasets.py -d ABCD BiTOD
parser = argparse.ArgumentParser(prog="Convert Dialogue Studio TOD datasets to unified format.")
parser.add_argument("-o", "--path-output", help="Folder to store all the converted datasets", default="./tod_datasets/")
parser.add_argument("-d", "--datasets",
                    nargs='*',
                    help="List of dataset names to convert (default: all datasets)",
                    default=["ABCD", "BiTOD", "Disambiguation", "DSTC2-Clean", "FRAMES", "GECOR", "HDSA-Dialog", "KETOD",
                             "MS-DC", "MulDoGO", "MULTIWOZ2_2", "MultiWOZ_2.1", "SGD", "SimJointMovie",
                             "SimJointRestaurant", "Taskmaster1", "Taskmaster2", "Taskmaster3", "WOZ2_0"])
args = parser.parse_args()


STR_USER = "user"
STR_SYSTEM = "system"
MIN_INTENT_WORDS = 4
MIN_TURN_WORDS = 2

RE_MSDC_ACT = r"(\w+)\((?:(\w+)(?:=([^;\)]+))?)?(.*)\)"
RE_MSDC_ACT_ARGS = r";(\w+)(?:=([^;]+))?"

INVALID_LABELS = ["other", "not sure", "switch frame"]

DA_NORMALIZED = {
    # INFORM
    # "inform": "inform",  # inform slot value

    "notify fail": "inform failure",
    "notify failure": "inform failure",
    "no result": "inform failure",
    "nobook": "inform failure",
    "nooffer": "inform failure",
    "sorry": "inform failure",
    "cant understand": "inform failure",
    "canthelp": "inform failure",
    "reject": "inform failure",

    "book": "inform success",
    "offerbooked": "inform success",
    "notify success": "inform success",

    # REQUEST
    # "request": "request",  # request slot value

    "request alt": "request alternative",

    "request compare": "request compare",

    "request update": "request update",

    "request more": "request more",
    "req more": "request more",
    "moreinfo": "request more",
    "hearmore": "request more",

    # CONFIRM
    # "confirm": "confirm",  # confirm slot value

    "confirm answer": "confirm answer",

    "confirm question": "confirm question",

    # AGREEMENT
    "affirm": "agreement",
    "affirm intent": "agreement",

    # DISAGREEMENT
    "negate": "disagreement",
    "negate intent": "disagreement",
    "deny": "disagreement",

    # OFFER
    # "offer": "offer",
    "select": "offer",
    "multiple choice": "offer",
    "offerbook": "offer",

    # RECOMMENDATION
    "suggest": "recommendation",
    "recommend": "recommendation",

    # GREETING
    # "greeting": "greeting",

    # THANK YOU
    # "thank you": "thank you",
    "thanks": "thank you",
    "thankyou": "thank you",
    "you are welcome": "thank you",
    "welcome": "thank you",

    # GOOD BYE
    # "good bye": "good bye",
    "goodbye": "good bye",
    "closing": "good bye",
}
DA_PARENT = {
    # INFORM
    "inform": "inform",
    "inform failure": "inform",
    "inform success": "inform",

    # REQUEST
    "request": "request",
    "request alternative": "request",
    "request compare": "request",
    "request update": "request",
    "request more": "request",

    # CONFIRM
    "confirm": "confirm",
    "confirm answer": "confirm",
    "confirm question": "confirm",
}

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()
def sorted_unique_output(func):
    def remove_duplicates_clean(values):
        values = [" ".join(lemmatizer.lemmatize(w.lower(), pos=wordnet.NOUN)
                           for w in re.findall(r"[a-zA-Z]+", e)
                           if len(w) > 1)
                  for e in dict.fromkeys(values).keys() if e]
        return [v for v in values if v and v not in INVALID_LABELS]

    def wrapper(*args, **kwargs):
        # remove duplicates but keep the order and slugify string values
        return (remove_duplicates_clean(v)
                if v and isinstance(v, list)
                else v
                for v in func(*args, **kwargs))
    return wrapper


def preprocess_turn_text(text):
    return text.strip().strip('"')


def parse_MSDC_act(text):
    m = re.match(RE_MSDC_ACT, text)
    slots = []
    act, slot, value = m.group(1), m.group(2), m.group(3)

    if slot:
        slots.append((slot, value))
        if m.group(4):
            slots.extend(re.findall(RE_MSDC_ACT_ARGS, m.group(4)))
    return act.lower(), slots


def get_turn(speaker, turn, dialog_acts=None, slots=None, intents=None, domains=None):
    # TODO: add label normalization "labels": {"dialog_acts": [], "parent_dialog_acts": []}, "raw_labels": {}
    turn = {
        "speaker": STR_USER if speaker == "user" else STR_SYSTEM,
        "text": preprocess_turn_text(turn["user utterance" if speaker == "user" else "system response" ]),
        "domains": domains,
        "labels": {}
    }
    if dialog_acts:
        normalized_acts = [DA_NORMALIZED[da] if da in DA_NORMALIZED else da
                           for da in dialog_acts]
        turn["labels"]["dialog_acts"] = {
                        "acts" : normalized_acts,
                        "main_acts" : [DA_PARENT[da] if da in DA_PARENT else da
                                       for da in normalized_acts],
                        "original_acts" : dialog_acts,
                    }
    if slots:
        turn["labels"]["slots"] = slots
    if intents:
        turn["labels"]["intents"] = intents
    return turn


@sorted_unique_output
def get_ABCD_acts_slots_intent(turn, speaker, dialog=None):
    slots, intent = [], None
    for seg in json.loads(turn[f"original {speaker} side information"])["delexed"]:
        seg_slots = re.findall(r"<[\w_]+>", seg["text"])
        if seg_slots:
            slots.extend(slot[1:-1] for slot in seg_slots)
        if speaker == "user" and intent is None and seg["targets"][0] and len(turn["user utterance"].split(" ")) >= MIN_INTENT_WORDS:
            intent = seg["targets"][0]

    return None, slots, [intent] if intent else intent, [json.loads(dialog["original dialog info"])["flow"]]


@sorted_unique_output
def get_BiTOD_acts_slots_intent(turn, speaker, dialog=None):
    def get_intent_parts(intent):
        return [s for s in re.split(r"(?:_en_US_)|(?:_en$)", intent) if s]
    acts, slots, intent = [], [], None
    for action in json.loads(turn[f"original {speaker} side information"])["Actions"]:
        if not intent and action["act"] == "inform_intent":
            if not re.search(r"(?:_en_US_)|(?:_en$)", action["value"][0]):
                raise ValueError("Non-English utterance found, turn:", turn)
            intent = "-".join(get_intent_parts(action["value"][0]))
        elif action["slot"] != "intent":
            acts.append(action["act"])
            if action["slot"]:
                slots.append(action["slot"])
    domain = get_intent_parts(json.loads(turn[f"original user side information"])["active_intent"])[0]
    return acts, slots, [intent] if intent else intent, [domain]


@sorted_unique_output
def get_Disambiguation_acts_slots_intents(turn, speaker, dialog=None):
    acts, slots, domains = [], [], []
    info = json.loads(turn[f"original {speaker} side information"])
    if "dialog_act" in info:
        for act, turn_slots in info["dialog_act"].items():
            turn_slots = [s[0].lower() for s in turn_slots if s[0] != "none"]
            if turn_slots:
                domain, act = act.lower().split("-")
                domains.append(domain)
                acts.append(act)
                slots.extend(turn_slots)
    return acts, slots, turn["intent"].split(", "), domains


@sorted_unique_output
def get_DSTC2_acts_slots_intents(turn, speaker, dialog=None):
    acts, slots = [], []
    info = json.loads(turn[f"original {speaker} side information"])

    if speaker == "user" and "turn_label" in info:
        for label, value in info["turn_label"]:
            if label  == "request":
                acts.append("request")
                slots.append(value)
            elif label:
                acts.append("inform")
                slots.append(label)
    elif speaker == "system" and "system_acts" in info:
        for slot in info["system_acts"]:
            if isinstance(slot, list):
                if slot[0]:
                    acts.append("inform")
                    slots.append(slot[0])
            elif slot:
                acts.append("request")
                slots.append(slot)

    return acts, slots, None, ["restaurant"]


@sorted_unique_output
def get_FRAMES_acts_slots_intents(turn, speaker, dialog=None):
    slots, intents = [], None
    info = json.loads(turn[f"original {speaker} side information"])

    acts = [info["da_label"]] if info["da_label"] and info["da_label"] != "null" else []
    if speaker == "user":
        slots.extend(iter(json.loads(turn[f"original system side information"])["slots"].keys()))
        if turn["dst"]:
            valid_acts = iter(json.loads(dialog["external knowledge non-flat"])["slots and values"].keys())
            valid_acts = '|'.join(f"(?:{act})" for act in valid_acts)
            acts.extend(set(re.findall(f"\\b{valid_acts}\\b", turn["dst"])))
        intents = acts

    return acts, slots, intents, ["travel"]


@sorted_unique_output
def get_GECOR_acts_slots_intents(turn, speaker, dialog=None):
    acts, slots = [], []
    info = json.loads(turn[f"original {speaker} side information"])

    if speaker == "user" and "slu" in info:
        turn_act_slots = [(act_slots["act"], act_slots["slots"])
                          for act_slots in info["slu"]]
        if turn["turn id"] > 0:
            pturn = dialog["log"][turn["turn id"] - 1]
            pinfo = json.loads(pturn[f"original user side information"])
            pturn_act_slots = [(act_slots["act"], act_slots["slots"])
                                for act_slots in pinfo["slu"]]
            turn_act_slots = [act_slots for act_slots in turn_act_slots if act_slots not in pturn_act_slots]

        for act, slot_value_pairs in turn_act_slots:
            acts.append(act)
            slots.extend(sv[0] if act != "request" and sv[0] != "slot" else sv[1] for sv in slot_value_pairs)
    elif speaker == "system" and "DA" in info:
        for slot in info["DA"]:
            if isinstance(slot, list):
                if slot[0]:
                    acts.append("inform")
                    slots.append(slot[0])
            elif slot:
                acts.append("request")
                slots.append(slot)

    return acts, slots, None, ["restaurant"]


@sorted_unique_output
def get_HDSA_acts_slots_intents(turn, speaker, dialog=None):
    acts, slots, domains = [], [], []
    info = json.loads(turn[f"original {speaker} side information"])

    if speaker == "user":
        # "user utterance" is populated with delexicalized version for some reason :/
        turn["user utterance"] = info["user_orig"]

        domain_slots = [(domain, slot_value) for domain in info["BS"] for slot_value in info["BS"][domain]]
        turn_ix = turn["turn id"] - 1
        if turn_ix > 0:
            pturn = dialog["log"][turn_ix - 1]
            pinfo = json.loads(pturn[f"original user side information"])
            pdomain_slots = [(domain, slot_value)
                             for domain in pinfo["BS"]
                             for slot_value in pinfo["BS"][domain]]
            domain_slots = [dom_slots for dom_slots in domain_slots if dom_slots not in pdomain_slots]

        for domain, slot_value in domain_slots:
            acts.append("inform")
            domains.append(domain)
            slots.append(slot_value[0])

    elif speaker == "system":
        # "system response" is populated with delexicalized version for some reason :/
        turn["system response"] = info["sys_orig"]
        dom_act_slots = [act.split('-') for act in info["act"] if info["act"][act] != "none"]

        for domain, act, slot in dom_act_slots:
            domains.append(domain)
            if act != "none":
                acts.append(act)
            if slot != "none":
                slots.append(slot)

    return acts, slots, None, domains


@sorted_unique_output
def get_KETOD_acts_slots_intents(turn, speaker, dialog=None):
    acts, slots, intents, domains = [], [], [], []
    info = json.loads(turn[f"original {speaker} side information"])

    if "original_utt" in info:
        turn["user utterance" if speaker == "user" else "system response"] = info["original_utt"]

    for frame in info["frames"]:
        domains.append(frame["service"].lower().split("_")[0])
        for action in frame["actions"]:
            if action["slot"] == "intent":
                intents.extend(action["values"])
            else:
                act = action["act"].lower()
                acts.append("inform" if "inform" in act else act)
                if action["slot"]:
                    slots.append(action["slot"].lower())

    return acts, slots, intents, domains


@sorted_unique_output
def get_MSDC_acts_slots_intents(turn, speaker, dialog=None):
    acts, slots = [], []
    msdc_acts = json.loads(turn[f"original {speaker} side information"])["act"]

    for msdc_act in msdc_acts:
        if msdc_act.strip():
            act, act_slots = parse_MSDC_act(msdc_act)

            acts.append(act)
            slots.extend(s[0] for s in act_slots if s[0] != act)
            # if at least one slot contains value, add "inform" as act too
            if "inform" not in acts and len([s[1] for s in act_slots if s[1]]) > 0:
                acts.append("inform")

    return acts, slots, turn["intent"].split(", "), [json.loads(dialog["original dialog info"])["domain"]]


@sorted_unique_output
def get_MulDoGO_acts_slots_intents(turn, speaker, dialog=None):
    info = json.loads(turn[f"original {speaker} side information"])

    if "slot-labels" not in info:
        return None, None, None, None

    slots = [slot
             for slots in info["slot-labels"]
             for slot in slots.split() if slot != "O"]
    return None, slots, info["intent"], [json.loads(dialog["original dialog info"])["domain"]]


@sorted_unique_output
def get_MULTIWOZ2_2_acts_slots_intents(turn, speaker, dialog=None):
    if speaker == "system":
        return None, None, None, None

    frames = json.loads(turn[f"original user side information"])["frames"]

    turn_slot_values = [slot_values
                        for frame in frames
                        for slot_values in frame["state"]["slot_values"].items()]
    turn_ix = turn["turn id"] - 1
    if turn_ix > 0:
        pturn = dialog["log"][turn_ix - 1]
        pframes = json.loads(pturn[f"original user side information"])["frames"]
        pturn_slot_values = [slot_values
                             for frame in pframes
                             for slot_values in frame["state"]["slot_values"].items()]
        turn_slot_values = [slot_values
                            for slot_values in turn_slot_values
                            if slot_values not in pturn_slot_values]
    domains, slots = [], []
    for slot, values in turn_slot_values:
        domain, slot = slot.split("-")
        domains.append(domain)
        slots.append(slot)
    acts = ["inform"] if slots else []

    request_slots = [slot.split("-")
                     for frame in frames
                     for slot in frame["state"]["requested_slots"]]
    if request_slots:
        acts.append("request")
        slots.extend(slot for _, slot in request_slots)
        domains.extend(domain for domain, _ in request_slots)
    if dialog["new dialog id"] == "MULTIWOZ2_2--train--2":
        pass
    return acts, slots, turn["intent"].split(", "), domains


@sorted_unique_output
def get_SGD_acts_slots_intents(turn, speaker, dialog=None):
    acts, slots, intents, domains = [], [], [], []
    frames = json.loads(turn[f"original {speaker} side information"])["frames"]

    for frame in frames:
        domains.append(frame["service"].lower().split("_")[0])
        for action in frame["actions"]:
            if action["slot"] == "intent":
                intents.extend(action["values"])
            else:
                act = action["act"].lower()
                acts.append("inform" if "inform" in act else act)
                if action["slot"]:
                    slots.append(action["slot"].lower())

    return acts, slots, intents, domains


@sorted_unique_output
def get_SimJointGEN_acts_slots_intents(turn, speaker, dialog=None):
    acts, slots = [], []
    info = json.loads(turn[f"original {speaker} side information"])

    if speaker == "user" and "dialog_state" in info:
        slot_values = info["dialog_state"].items()
        turn_ix = turn["turn id"] - 1
        if turn_ix > 0:
            pturn = dialog["log"][turn_ix - 1]
            pinfo = json.loads(pturn[f"original user side information"])
            if "dialog_state" in pinfo:
                pslot_values = pinfo["dialog_state"].items()
                slot_values = [slot_value for slot_value in slot_values if slot_value not in pslot_values]

        for slot, value in slot_values:
            acts.append("inform")
            slots.append(slot)

    elif speaker == "system" and "system_acts" in info:
        for system_act in info["system_acts"]:
            acts.append(system_act["name"])
            if "slot_values" in system_act:
                slots.extend(slot for slot in system_act["slot_values"])

    return acts, slots, None, ["movie"]


@sorted_unique_output
def get_SimJointMovie_acts_slots_intents(turn, speaker, dialog=None):
    acts, slots, intents = [], [], None
    info = json.loads(turn[f"original {speaker} side information"])

    if speaker == "user":
        if "user_intents" in info and info["user_intents"]:
            intents = info["user_intents"]
        if "user_acts" in info:
            acts.extend(act["type"] for act in info["user_acts"])
        if "user_utterance" in info and "slots" in info["user_utterance"]:
            slots.extend(slot["slot"] for slot in info["user_utterance"]["slots"] if slot["slot"])

    elif speaker == "system" and "system_acts" in info:
        acts.extend(system_act["type"] for system_act in info["system_acts"])
        slots.extend(system_act["slot"] for system_act in info["system_acts"] if "slot" in system_act)

    return acts, slots, intents, ["movie" if "movie" in dialog["new dialog id"].lower() else "restaurant"]


@sorted_unique_output
def get_Taskmaster1_acts_slots_intents(turn, speaker, dialog=None):
    info = json.loads(turn[f"original {speaker} side information"])

    slots, domains = [], []
    if "segments" in info and info["segments"]:
        for segment in info["segments"]:
            if "annotations" in segment:
                domains.extend(an["name"].split(".")[0].split("_")[0] for an in segment["annotations"])
                slots.extend(".".join(an["name"].split(".")[1:3]) for an in segment["annotations"])

    return ["inform"] if slots else None, slots, None, domains


@sorted_unique_output
def get_Taskmaster3_acts_slots_intents(turn, speaker, dialog=None):
    info = json.loads(turn[f"original {speaker} side information"])

    slots = []
    if "segments" in info and info["segments"]:
        for segment in info["segments"]:
            if "annotations" in segment:
                slots.extend(".".join(an["name"].split(".")[:2]) for an in segment["annotations"])

    return ["inform"] if slots else None, slots, None, ["movie"]



get_acts_slots_intent = {
    "ABCD": get_ABCD_acts_slots_intent,
    "BiTOD": get_BiTOD_acts_slots_intent,
    "Disambiguation": get_Disambiguation_acts_slots_intents,
    "DSTC2-Clean": get_DSTC2_acts_slots_intents,
    "FRAMES": get_FRAMES_acts_slots_intents,
    "GECOR": get_GECOR_acts_slots_intents,
    "HDSA-Dialog": get_HDSA_acts_slots_intents,
    "KETOD": get_KETOD_acts_slots_intents,
    "MS-DC": get_MSDC_acts_slots_intents,
    "MulDoGO": get_MulDoGO_acts_slots_intents,
    "MULTIWOZ2_2": get_MULTIWOZ2_2_acts_slots_intents,
    "MultiWOZ_2.1": get_Disambiguation_acts_slots_intents,
    "SGD": get_SGD_acts_slots_intents,
    "SimJointGEN": get_SimJointGEN_acts_slots_intents,
    "SimJointMovie": get_SimJointMovie_acts_slots_intents,
    "SimJointRestaurant": get_SimJointMovie_acts_slots_intents,
    "Taskmaster1": get_Taskmaster1_acts_slots_intents,
    "Taskmaster2": get_Taskmaster1_acts_slots_intents,
    "Taskmaster3": get_Taskmaster3_acts_slots_intents,
    "WOZ2_0": get_DSTC2_acts_slots_intents
}


if __name__ == "__main__":
    login()
    for dataset in tqdm(args.datasets, desc="Converting datasets"):
        data = load_dataset('Salesforce/dialogstudio', dataset, trust_remote_code=True)

        path_output = os.path.join(args.path_output, dataset)
        os.makedirs(path_output, exist_ok=True)
        new_dataset = {
            "stats": {
                "domains": Counter(),
                "labels": {
                    "dialog_acts": {
                        "acts" : Counter(),
                        "main_acts" : Counter(),
                        "original_acts" : Counter(),
                    },
                    "slots": Counter(),
                    "intents": Counter(),
                }
            },
            "dialogs": {}
        }
        stats = new_dataset["stats"]
        for split in data:
            for dialog in data[split]:
                turns = []
                prev_intents = None
                for turn in dialog["log"]:
                    for speaker in ["user", "system"]:
                        acts, slots, intents, doms = get_acts_slots_intent[dataset](turn, speaker, dialog)

                        if intents:
                            if intents == prev_intents:
                                intents = None
                            else:
                                prev_intents = intents

                        if acts or slots or intents:
                            turn_formatted = get_turn(speaker, turn, acts, slots, intents, doms)
                            if len(turn_formatted["text"].split()) >= MIN_TURN_WORDS:
                                turns.append(turn_formatted)

                            stats["domains"].update(doms)
                            if acts:
                                for acts_field in ["acts", "main_acts", "original_acts"]:
                                    stats["labels"]["dialog_acts"][acts_field].update(
                                        turn_formatted["labels"]["dialog_acts"][acts_field]
                                    )
                            if slots:
                                stats["labels"]["slots"].update(slots)
                            if intents:
                                stats["labels"]["intents"].update(intents)
                if turns:
                    new_dataset["dialogs"][dialog["new dialog id"]] = turns

        with open(os.path.join(path_output, "data.json"), "w") as writer:
            json.dump(new_dataset, writer)
        with open(os.path.join(path_output, "stats.json"), "w") as writer:
            json.dump(stats, writer)

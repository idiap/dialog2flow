# -*- coding: utf-8 -*-
"""
Script to convert original SpokenWOZ dataset to ground truth trajectories
json format with standardized dialog act annotation.

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import re
import json
import argparse


DEFAULT_TOKEN_START = "[start]"
DEFAULT_TOKEN_END = "[end]"

# python groundtruth_trajectories.py -i "data/spokenwoz/"
parser = argparse.ArgumentParser(prog="Convert SpokenWOZ dataset to trajectories (normalized turns as 'SPEAKER: domain-act slot')")
parser.add_argument("-i", "--input-path", help="Path to SpokenWOZ dataset containing the original 'text_5700_train_dev' and 'text_5700_test' folders", required=True)
args = parser.parse_args()


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


def preprocess(text):
    text = re.sub(r"(\w)\s+'\s*", r"\1'", text)
    text = re.sub(r"(\w)\s+\?", r"\1'", text)
    text = " ".join(re.findall(r"[a-zA-Z0-9'?]+", text))
    return text.lower()


def get_turn(turn, act_slots):
    return {"tag": turn['tag'],
            "text": preprocess(turn['text']),
            "turn": f"{turn['tag'].upper()}: {act_slots}".strip()}  # SPK: domain-act slot*


def generate_trajectories(data, only_single_domain=False):
    print("Generating trajectories%s..." % (" (only for single domain calls)" if only_single_domain else ""))
    trajectories = {}
    for dialog_id in data:
        if only_single_domain and not dialog_id.startswith("SNG"):
            continue

        trajectories[dialog_id] = {"goal": {g:v for g, v in data[dialog_id]["goal"].items() if v and g != "profile"},
                                   "log": [{"tag": None, "text": None, "turn": DEFAULT_TOKEN_START}]}

        for turn in data[dialog_id]["log"]:
            act_slots = []
            for act in turn["dialog_act"]:

                if re.match(r"^\w+-(\w)", act):
                    domain, dialog_act = act.split("-")
                else:
                    dialog_act = act

                if dialog_act in ["ack", "wait", "backchannel"]:
                    continue

                if dialog_act in DA_NORMALIZED:
                    dialog_act = DA_NORMALIZED[dialog_act]

                slots = set()
                for slot in turn["dialog_act"][act]:
                    slot = "" if slot[0] == "none" else slot[0]
                    if dialog_act == "ack":
                        slot = ""
                    if slot:
                        slots.add(slot)
                slots = " ".join(sorted(list(slots)))
                act_slots.append(f"{dialog_act} {slots}".strip())

            # Skipping turns with multiple acts
            if len(act_slots) > 1:
                continue

            if act_slots:
                trajectories[dialog_id]["log"].append(get_turn(turn, "; ".join(act_slots)))

        trajectories[dialog_id]["log"].append({"tag": None, "text": None, "turn": DEFAULT_TOKEN_END})
    print(f"Finished.")
    return trajectories


if __name__ == "__main__":
    all_trajectories = {}
    domains = {}
    path_spokenwoz_train = os.path.join(args.input_path, "text_5700_train_dev/data.json")
    path_spokenwoz_test = os.path.join(args.input_path, "text_5700_test/data.json")
    for path in [path_spokenwoz_train, path_spokenwoz_test]:
        print(f"\n> About to process '{path}'")
        with open(path) as reader:
            data = json.load(reader)
        for only_single_domain in (False, True):
            prefix = ".single_domain" if only_single_domain else ""
            path_output = f"{os.path.split(path)[0]}/trajectories{prefix}.json"
            with open(path_output, "w") as writer:
                trajectories = generate_trajectories(data, only_single_domain)
                json.dump(trajectories, writer)
                print(f"  * {len(trajectories)} trajectories saved in '{path_output}'")

            if only_single_domain:
                for call_id, trajectory in trajectories.items():
                    assert len(trajectory["goal"]) == 1, f"Call {call_id} is marked as single-domain but contains {len(trajectory['goal'])} domains"
                    domain = next(iter(trajectory["goal"]))
                    if domain not in domains:
                        domains[domain] = []
                    domains[domain].append(call_id)

            if prefix not in all_trajectories:
                all_trajectories[prefix] = {}
            all_trajectories[prefix].update(trajectories)

    print("\n> Generating trajectories for whole corpus (train + test)...")
    for prefix in all_trajectories:
        print("Multidomain calls..." if prefix else "Single-domain calls only...")
        path_base = os.path.split(path_spokenwoz_train)[0]
        path_output = f"{os.path.split(path_base)[0]}/trajectories{prefix}.json"
        with open(path_output, "w") as writer:
            json.dump(all_trajectories[prefix], writer)
            print(f"  * {len(all_trajectories[prefix])} trajectories saved in '{path_output}'")

    print("\n> Saving domains information file...")
    path_output = f"{os.path.split(path_base)[0]}/domains.json"
    with open(path_output, "w") as writer:
        json.dump(domains, writer)
        print(f"  * {len(domains)} domains saved in '{path_output}'\n")

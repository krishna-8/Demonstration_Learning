# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import numpy as np
import copy
from functools import cmp_to_key

class Voice():
    def __init__(self, use_synonyms=True, load=True):
        self.use_syn = use_synonyms
        if load:
            self.dict = self._loadDictionary("../GDrive/glove.6B.50d.txt")
            self.inv_dict = {v: k for (k, v) in self.dict.items()}

        self.map_bowls          = {}
        self.map_bowls[1]       = ("yellow", "small", "round")
        self.map_bowls[2]       = ("red",    "small", "round")
        self.map_bowls[3]       = ("green",  "small", "round")
        self.map_bowls[4]       = ("blue",   "small", "round")
        self.map_bowls[5]       = ("pink",   "small", "round")
        self.map_bowls[6]       = ("yellow", "large", "round")
        self.map_bowls[7]       = ("red",    "large", "round")
        self.map_bowls[8]       = ("green",  "large", "round")
        self.map_bowls[9]       = ("blue",   "large", "round")
        self.map_bowls[10]      = ("pink",   "large", "round")
        self.map_bowls[11]      = ("yellow", "small", "square")
        self.map_bowls[12]      = ("red",    "small", "square")
        self.map_bowls[13]      = ("green",  "small", "square")
        self.map_bowls[14]      = ("blue",   "small", "square")
        self.map_bowls[15]      = ("pink",   "small", "square")
        self.map_bowls[16]      = ("yellow", "large", "square")
        self.map_bowls[17]      = ("red",    "large", "square")
        self.map_bowls[18]      = ("green",  "large", "square")
        self.map_bowls[19]      = ("blue",   "large", "square")
        self.map_bowls[20]      = ("pink",   "large", "square")

        self.map_cups           = {}
        self.map_cups[1]        = ("red")
        self.map_cups[2]        = ("green")
        self.map_cups[3]        = ("blue")

        self.synonyms = {}
        self.synonyms["round"]  = ["round", "curved"]
        self.synonyms["square"] = ["square", "rectangular"]
        self.synonyms["small"]  = ["small", "tiny", "smallest", "petite", "meager"]
        self.synonyms["large"]  = ["large", "largest", "big", "biggest", "giant", "grand"]
        self.synonyms["red"]    = ["red", "ruby", "cardinal",   "crimson", "maroon", "carmine"]
        self.synonyms["green"]  = ["green", "olive", "jade"]
        self.synonyms["blue"]   = ["blue", "azure", "cobalt", "indigo"]
        self.synonyms["yellow"] = ["yellow", "amber", "bisque", "blond", "gold", "golden"]
        self.synonyms["pink"]   = ["pink", "salmon", "rose"]
        self.synonyms["cup"]    = ["cup", "container", "grail", "stein"]
        self.synonyms["pick"]   = ["pick_up", "gather", "take_up", "grasp", "elevate", "lift", "raise", "lift_up", "grab"] 
        self.synonyms["pour"]   = ["pour", "spill", "fill"]
        self.synonyms["put"]   = ["put", "drop"]
        self.synonyms["little"] = ["a little", "some", "a small amount"]
        self.synonyms["much"]   = ["everything", "all of it", "a lot"]
        self.synonyms["goto"]   = ["go_to", "move_to", "advance_to", "progress_to", "carry_to", "transport_to", "transfer_to"]
        self.synonyms["bowl"]   = ["bowl", "basin", "dish", "pot"]
        self.synonyms["left"]   = ["left", "port"]
        self.synonyms["right"]  = ["right", "starboard"]

        self.test_words = {}
        self.test_words["round"]  = "circular"
        self.test_words["square"] = "quadrilateral"
        self.test_words["small"]  = "little"
        self.test_words["large"]  = "huge"
        self.test_words["red"]    = "scarlet"
        self.test_words["green"]  = "emerald"
        self.test_words["blue"]   = "beryl"
        self.test_words["yellow"] = "lemon"
        self.test_words["pink"]   = "rosy"
        self.test_words["pick"]   = "collect"
        self.test_words["pour"]   = "transfer"
        self.test_words["put"]   = "release"

        self.test_words["little"] = "a bit"
        self.test_words["much"]   = "all"
        
        self.test_words["bowl"]   = "bowl" #
        self.test_words["cup"]    = "container" #

        self.templates = [
            [ "{pick} the {cup_description} {cup} {pick_support}", "{pick} the {cup} {cup_description} {pick_support}"  ],
            [ "{pour} {amount} into the {bowl_description} {bowl}", "{pour} {amount} into the {bowl_description}" ]
            # [ "{put} on the {bowl_description} {bowl}", "{put} on the {bowl_description} {bowl}" ]
        ]

        # Make adjustments in case we don't want synonyms
        if not self.use_syn:
            for k, v in self.synonyms.items():
                self.synonyms[k] = [k]        

        # Synonym-reverse Dict:
        self.basewords = {}
        for k, v in self.synonyms.items():    
            for w in v:
                self.basewords[w] = k

        self.word_list = [syn for key in self.synonyms.keys() for syn in self.synonyms[key]]

    def createTestSentence(self, voice):
        words   = voice.split(" ")
        new_stn = []

        replace = []
        for phrase in self.word_list:
            base = self.basewords[phrase]
            if phrase in voice and base in self.test_words.keys():
                base = self.test_words[base]
                if type(base) == list:
                    base = np.random.choice(base)
                replace.append((phrase, base))        
        for rep in replace:
            voice = voice.replace(rep[0], rep[1])
        return voice

    def _loadDictionary(self, file):
        __dictionary = {}
        __dictionary[""] = 0 # Empty string
        fh = open(file, "r", encoding="utf-8")
        for line in fh:
            if len(__dictionary) >= 300000:
                break
            tokens = line.strip().split(" ")
            __dictionary[tokens[0]] = len(__dictionary)
        fh.close()
        return __dictionary
    
    def tokensToSentence(self, tokens):
        words = [self.inv_dict[token] for token in tokens]
        return " ".join(words)

    def getMinimalBowlDescription(self, task):
        color, size, shape = self.map_bowls[task["target/id"]]
        other_bowls        = task["ints"][2:2+task["ints"][0]]
        other_bowls        = [self.map_bowls[b] for b in other_bowls if b != task["target/id"]]

        # If we don't want synonyms, limit our self to colors only
        if not self.use_syn:
            index = 0 # 0: color, 1: size, 2: shape
            return np.random.choice(self.synonyms[self.map_bowls[task["target/id"]][index]])
        
        # Unique properties of rank 0:
        if len(other_bowls) == 0:
            return ""

        if np.random.choice([True, False]):
            one_color   = len([i for i in range(len(other_bowls)) if other_bowls[i][0] == color]) == 1
            one_size    = len([i for i in range(len(other_bowls)) if other_bowls[i][1] == size])  == 1
            one_shape   = len([i for i in range(len(other_bowls)) if other_bowls[i][2] == shape]) == 1
            r1_unique     = np.asarray([one_color, one_shape, one_size])

            one_color_size  = len([i for i in range(len(other_bowls)) if other_bowls[i][0] == color and other_bowls[i][1] == size])  == 1
            one_color_shape = len([i for i in range(len(other_bowls)) if other_bowls[i][0] == color and other_bowls[i][2] == shape]) == 1
            one_size_shape  = len([i for i in range(len(other_bowls)) if other_bowls[i][1] == size  and other_bowls[i][2] == shape]) == 1
            r2_unique     = np.asarray([one_color_size, one_color_shape, one_size_shape])

            if np.any(r2_unique):
                indices = np.argwhere(r2_unique == True)[:,0]
                index   = np.random.choice(indices)
                if index == 0:
                    # size color
                    other_bowl = [other_bowls[i][2] for i in range(len(other_bowls)) if other_bowls[i][0] == color and other_bowls[i][1] == size]
                    return "0" + np.random.choice(self.synonyms[size]) + " " + np.random.choice(self.synonyms[color]) + " " + np.random.choice(self.synonyms["bowl"]) + " which is not " + other_bowl[0]
                elif index == 1:
                    # shape color
                    other_bowl = [other_bowls[i][1] for i in range(len(other_bowls)) if other_bowls[i][0] == color and other_bowls[i][2] == shape]
                    return "0" + np.random.choice(self.synonyms[shape]) + " " + np.random.choice(self.synonyms[color]) + " " + np.random.choice(self.synonyms["bowl"]) + " which is not " + other_bowl[0]
                elif index == 2:
                    # size shape
                    other_bowl = [other_bowls[i][0] for i in range(len(other_bowls)) if other_bowls[i][1] == size  and other_bowls[i][2] == shape]
                    return "0" + np.random.choice(self.synonyms[size]) + " " + np.random.choice(self.synonyms[shape]) + " " + np.random.choice(self.synonyms["bowl"]) + " which is not " + other_bowl[0]


        # Unique properties of rank 1:
        u_color   = len([i for i in range(len(other_bowls)) if other_bowls[i][0] == color]) == 0
        u_size    = len([i for i in range(len(other_bowls)) if other_bowls[i][1] == size])  == 0
        u_shape   = len([i for i in range(len(other_bowls)) if other_bowls[i][2] == shape]) == 0
        r1_unique = np.asarray([u_color, u_size, u_shape])

        if np.any(r1_unique):
            indices = np.argwhere(r1_unique == True)[:,0]
            index   = np.random.choice(indices)
            return np.random.choice(self.synonyms[self.map_bowls[task["target/id"]][index]])

        # Unique properties of rank 2:
        u_color_size  = len([i for i in range(len(other_bowls)) if other_bowls[i][0] == color and other_bowls[i][1] == size])  == 0
        u_color_shape = len([i for i in range(len(other_bowls)) if other_bowls[i][0] == color and other_bowls[i][2] == shape]) == 0
        u_size_shape  = len([i for i in range(len(other_bowls)) if other_bowls[i][1] == size  and other_bowls[i][2] == shape]) == 0
        r2_unique     = np.asarray([u_color_size, u_color_shape, u_size_shape])

        if np.any(r2_unique):
            indices = np.argwhere(r2_unique == True)[:,0]
            index   = np.random.choice(indices)
            if index == 0:
                # size color
                return np.random.choice(self.synonyms[size]) + " " + np.random.choice(self.synonyms[color])
            elif index == 1:
                # shape color
                return np.random.choice(self.synonyms[shape]) + " " + np.random.choice(self.synonyms[color])
            elif index == 2:
                # size shape
                return np.random.choice(self.synonyms[size]) + " " + np.random.choice(self.synonyms[shape])

        # Unique properties of rank 3:
        return np.random.choice(self.synonyms[size]) + " " + np.random.choice(self.synonyms[shape]) + " " + np.random.choice(self.synonyms[color])
        
    def getMinimalCupDescription(self, task, alt):
        def compare(item1, item2):
            if item1[1] < item2[1]:
                return -1
            elif item1[1] > item2[1]:
                return 1
            else:
                if item1[2] < item2[2]:
                    return -1
                elif item1[2] > item2[2]:
                    return 1
        color = np.random.choice(self.synonyms[self.map_cups[task["target/id"]]])
        ints = task["ints"]
        nbowls = ints[0]
        ncups = ints[1]
        cup_ids=ints[2+nbowls:2+nbowls+ncups]
        cups_pos = []
        for i in cup_ids:
            index = np.argwhere(np.asarray(cup_ids)==i)[0,0] + nbowls
            cups_pos.append((i, task["floats"][index * 3],  task["floats"][index * 3 + 1]))
        cups_pos = sorted(cups_pos, key=cmp_to_key(compare))


        if alt and ncups > 1:
            other_cups = [np.random.choice(self.synonyms[self.map_cups[b]]) for b in cup_ids if b != task["target/id"]]
            if ncups == 2:
                single_other = "which is not " + np.random.choice(other_cups)
                return single_other
            else:
                multi_other = "which is neither " + other_cups[0] + " nor " + other_cups[1]
                return multi_other
        if task["ints"][1] == 1 and self.use_syn:
            return ""
        return color

    def generateSentence(self, task):
        pick             = np.random.choice(self.synonyms["pick"])
        alt = np.random.choice([True,False])
        # print(task["floats"])
        # print(task["ints"])
        pick_support     = "" if len(pick.split("_")) == 1 else pick.split("_")[1]
        pick             = pick.split("_")[0]
        amount           = np.random.choice(self.synonyms["little"]) if task["amount"] < 150 else np.random.choice(self.synonyms["much"])
        cup_description  = "" if task["target/type"] == "bowl" else self.getMinimalCupDescription(task, alt)
        bowl_description = "" if task["target/type"] == "cup"  else self.getMinimalBowlDescription(task)
        alt2 = False
        if bowl_description and bowl_description[0] == "0":
            alt2 = True
            bowl_description = bowl_description[1:]
        namespace        = {
            "pick": pick,
            "cup": np.random.choice(self.synonyms["cup"]), 
            "cup_description": cup_description, 
            "pick_support": pick_support, 
            "pour": np.random.choice(self.synonyms["pour"]),
            # "put": np.random.choice(self.synonyms["put"]), 
            "amount": amount, 
            "bowl_description": bowl_description,
            "bowl": np.random.choice(self.synonyms["bowl"])
        }
        if alt and task["ints"][task["phase"]] > 1:
            sentence = self.templates[task["phase"]][1]
        elif alt2 and task["phase"] == 1:
            sentence = self.templates[task["phase"]][1]
        else:
            sentence = self.templates[task["phase"]][0]
        sentence = sentence.format(**namespace)
        sentence = sentence.replace("  ", " ")
        return sentence


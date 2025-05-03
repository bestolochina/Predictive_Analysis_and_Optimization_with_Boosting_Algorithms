from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class BoostingTest(StageTest):
    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=15000)]

    def check(self, reply: str, attach):
        if not os.path.exists('../data'):
            return CheckResult.wrong("There is no directory called data")

        if 'insurance.csv' not in os.listdir('../data'):
            return CheckResult.wrong("There is no file called insurance.csv")

        if not reply:
            return CheckResult.wrong("Do not forget to print")

        reply = reply.strip().lower()

        if not (reply.startswith('{') and reply.endswith('}')):
            return CheckResult.wrong("Print the result as a dictionary")

        dict_pattern = r"^{(.+)}$"
        dict_match = re.search(dict_pattern, reply)

        if not dict_match:
            return CheckResult.wrong("Do not print an empty dictionary")

        if "numerical" not in reply:
            return CheckResult.wrong("The dictionary should contain a numerical key")

        if "categorical" not in reply:
            return CheckResult.wrong("The dictionary should contain a categorical key")

        if "shape" not in reply:
            return CheckResult.wrong("The dictionary should contain a shape key")

        num_pattern = r"'numerical': \[(.*?)\]"
        match_num = re.search(num_pattern, reply)

        if not match_num:
            return CheckResult.wrong("The numerical value should be a list of strings")

        cat_pattern = r"'categorical': \[(.*?)\]"
        match_cat = re.search(cat_pattern, reply)

        if not match_cat:
            return CheckResult.wrong("The categorical value should be a list of strings")

        shape_pattern = r"'shape': \[(.*?)\]"
        match_shape = re.search(shape_pattern, reply)

        if not match_shape:
            return CheckResult.wrong("The shape value should be a list of integers")

        num_value = sorted(match_num.group(1).split(", "))
        num_value = [v.strip("'") for v in num_value]

        if len(num_value) != 4:
            return CheckResult.wrong("There should be four items in the numerical list")

        if num_value != ['age', 'bmi', 'charges', 'children']:
            return CheckResult.wrong("One or more of the numerical features are incorrect")

        cat_value = sorted(match_cat.group(1).split(", "))
        cat_value = [v.strip("'") for v in cat_value]

        if len(cat_value) != 3:
            return CheckResult.wrong("There should be three items in the categorical list")

        if cat_value != ["region", 'sex', 'smoker']:
            return CheckResult.wrong("One or more of the categorical features are incorrect")

        shape_value = match_shape.group(1).split(", ")

        if len(shape_value) != 2:
            return CheckResult.wrong("There should be two items in the shape list: [nrows, ncols]")

        if not shape_value[0].isdigit() or not shape_value[1].isdigit():
            return CheckResult.wrong("The shape should be integer")

        shape_value = [int(v) for v in shape_value]

        if shape_value[0] != 1337:
            return CheckResult.wrong("The number of rows is incorrect. Did you drop duplicates?")

        if shape_value[1] != 7:
            return CheckResult.wrong("The number of columns are incorrect")

        return CheckResult.correct()
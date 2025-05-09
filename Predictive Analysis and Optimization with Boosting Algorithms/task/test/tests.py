from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import os
import re


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

        if 'train' not in reply:
            return CheckResult.wrong("The key 'train' is not in the dictionary")

        if 'validation' not in reply:
            return CheckResult.wrong("The key 'validation' is not in the dictionary")

        if 'test' not in reply:
            return CheckResult.wrong("The key 'test' is not in the dictionary")

        pattern = r"'(train|validation|test)': \[(.*?)\]"

        matched = re.findall(pattern, reply)

        if not matched or len(matched) != 3:
            return CheckResult.wrong("Ensure that the correct keys are in the dictionary")

        reply_dict = {}

        for one in matched:
            if len(one) != 2:
                return CheckResult.wrong("The value of a key is a list of tuples: [(nrow, ncol), (nrow,)]")

        for key, value in matched:
            reply_dict[key] = value

        train_value = reply_dict['train']

        pattern_num = r'\d+, ?\d+'

        train_match = re.findall(pattern_num, train_value)

        train_result = [
            int(match.strip()) for match in train_match[0].split(',') if match.strip().isdigit()
        ]

        if train_result[0] != 851:
            return CheckResult.wrong("The number of rows in train is incorrect")

        if train_result[-1] != 11:
            return CheckResult.wrong("The number of columns in train is incorrect")

        valid_value = reply_dict['validation']

        valid_match = re.findall(pattern_num, valid_value)

        valid_result = [
            int(match.strip()) for match in valid_match[0].split(',') if match.strip().isdigit()
        ]

        if valid_result[0] != 213:
            return CheckResult.wrong("The number of rows in validation is incorrect")

        if valid_result[-1] != 11:
            return CheckResult.wrong("The number of columns in validation is incorrect")

        test_value = reply_dict['test']

        test_match = re.findall(pattern_num, test_value)

        test_result = [
            int(match.strip()) for match in test_match[0].split(',') if match.strip().isdigit()
        ]

        if test_result[0] != 266:
            return CheckResult.wrong("The number of rows in test is incorrect")

        if test_result[-1] != 11:
            return CheckResult.wrong("The number of columns in test is incorrect")

        return CheckResult.correct()




"""This a reviewed implementation of the loading function found in:
https://github.com/jeniyat/StackOverflowNER/blob/master/code/DataReader/loader_so.py
"""

from itertools import chain


def merge_labels(input_file: str):
    """Merge similar labels for simpler experimentation."""

    merging_dict = {}

    merging_dict["Library_Function"] = "Function"
    merging_dict["Function_Name"] = "Function"

    merging_dict["Class_Name"] = "Class"
    merging_dict["Library_Class"] = "Class"

    merging_dict["Library_Variable"] = "Variable"
    merging_dict["Variable_Name"] = "Variable"

    merging_dict["Website"] = "Website"
    merging_dict["Organization"] = "Website"

    f_in = open(input_file, "r", encoding="utf-8")

    modified_file = input_file[:-4] + "_merged_labels.txt"
    f_out = open(modified_file, "w", encoding="utf-8")
    line_count = 0

    for line in f_in:
        line_count += 1
        line_values = line.strip().split()
        if len(line_values) < 2:
            opline = line
            f_out.write(opline)
            continue

        gold_word = line_values[0]
        gold_label = line_values[1]
        raw_word = line_values[2]
        raw_label = line_values[3]

        if gold_word != raw_word:
            print("wrong mapping:", line)

        word = gold_word
        label = gold_label

        if label == "O":
            opline = line
            f_out.write(opline)
            continue

        label_split = label.split("-", 1)

        label_prefix = label_split[0]
        label_name = label_split[1]

        if label_name in merging_dict:
            label_name = merging_dict[label_name]

        new_label = label_prefix + "-" + label_name
        opline = word + " " + new_label + " " + raw_word + " " + raw_label + "\n"
        f_out.write(opline)

    f_out.close()

    return modified_file


def load_sentences(path, merge_tag=True, replace_low_freq_tags=True):
    """Main function to load the tokenized sentences with the respective labels from a file path"""

    if merge_tag:
        path = merge_labels(path)

    set_of_selected_tags = []

    if replace_low_freq_tags:
        sorted_entity_list = ["Class", "Class_Name", "Library_Class", "Application",
                              "Library_Variable", "Variable_Name", "Variable",
                              "User_Interface_Element", "Code_Block", "Library_Function",
                              "Function_Name", "Function", "Language", "Library", "Data_Structure",
                              "Data_Type", "File_Type", "File_Name", "Version", "HTML_XML_Tag",
                              "Device", "Operating_System", "User_Name", "Website", "Output_Block",
                              "Error_Name", "Algorithm", "Organization", "Keyboard_IP", "Licence",
                              "Organization"
                              ]
        set_of_selected_tags.extend(sorted_entity_list[0:-6])
        if "Algorithm" not in set_of_selected_tags:
            set_of_selected_tags.append("Algorithm")

    sentence = []  # List of words in the current sentence
    sentences = []  # List of sentences
    count_question = 0
    count_answer = 0
    max_len = 0

    for line in open(path, "r", encoding="utf-8"):
        if line.startswith("Question_ID"):
            count_question += 1

        if line.startswith("Answer_to_Question_ID"):
            count_answer += 1

        if line.strip() == "":
            if len(sentence) > 0:
                output_line = " ".join(w[0] for w in sentence)

                if "code omitted for annotation" in output_line and "CODE_BLOCK :" in output_line:
                    sentence = []
                    continue
                elif "omitted for annotation" in output_line and "OP_BLOCK :" in output_line:
                    sentence = []
                    continue
                elif "Question_URL :" in output_line:
                    sentence = []
                    continue
                elif "Question_ID :" in output_line:
                    sentence = []
                    continue
                else:
                    sentences.append(sentence)
                    if len(sentence) > max_len:
                        max_len = len(sentence)
                    sentence = []
        else:
            line_values = line.strip().split()
            gold_word = line_values[0]
            gold_label = line_values[1]
            # raw_word = line_values[2]
            # raw_label = line_values[3]

            gold_word = " ".join(gold_word.split('-----'))

            gold_label_name = gold_label.replace("B-", "").replace("I-", "")

            if gold_label_name not in set_of_selected_tags:
                gold_label = "O"

            word_info = [gold_word, gold_label]
            sentence.append(word_info)

    print("------------------------------------------------------------")
    print("Number of questions in ", path, " : ", count_question)
    print("Number of answers in ", path, " : ", count_answer)
    print("Number of sentences in ", path, " : ", len(sentences))

    return sentences


def load_posts(path, merge_tag=True, replace_low_freq_tags=True):
    """Main function to load the tokenized posts with the respective labels from a file path"""

    if merge_tag:
        path = merge_labels(path)

    set_of_selected_tags = []

    if replace_low_freq_tags:
        sorted_entity_list = ["Class", "Class_Name", "Library_Class", "Application",
                              "Library_Variable", "Variable_Name", "Variable",
                              "User_Interface_Element", "Code_Block", "Library_Function",
                              "Function_Name", "Function", "Language", "Library", "Data_Structure",
                              "Data_Type", "File_Type", "File_Name", "Version", "HTML_XML_Tag",
                              "Device", "Operating_System", "User_Name", "Website", "Output_Block",
                              "Error_Name", "Algorithm", "Organization", "Keyboard_IP", "Licence",
                              "Organization"
                              ]
        set_of_selected_tags.extend(sorted_entity_list[0:-6])
        if "Algorithm" not in set_of_selected_tags:
            set_of_selected_tags.append("Algorithm")

    posts = []
    sentence = []  # List of words in the current sentence
    sentences = []  # List of sentences
    count_posts = 0
    max_len = 0

    for line in open(path, "r", encoding="utf-8"):
        if line.startswith("Question_ID") or line.startswith("Answer_to_Question_ID"):
            count_posts += 1
            if len(sentences) > 0:
                posts.append(list(chain.from_iterable(sentences)))
                sentences = []

        if line.strip() == "":
            if len(sentence) > 0:
                output_line = " ".join(w[0] for w in sentence)

                if "code omitted for annotation" in output_line and "CODE_BLOCK :" in output_line:
                    sentence = []
                    continue
                elif "omitted for annotation" in output_line and "OP_BLOCK :" in output_line:
                    sentence = []
                    continue
                elif "Question_URL :" in output_line:
                    sentence = []
                    continue
                elif "Question_ID :" in output_line:
                    sentence = []
                    continue
                else:
                    sentences.append(sentence)
                    if len(sentence) > max_len:
                        max_len = len(sentence)
                    sentence = []
        else:
            line_values = line.strip().split()
            gold_word = line_values[0]
            gold_label = line_values[1]

            gold_word = " ".join(gold_word.split('-----'))

            gold_label_name = gold_label.replace("B-", "").replace("I-", "")

            if gold_label_name not in set_of_selected_tags:
                gold_label = "O"

            word_info = [gold_word, gold_label]
            sentence.append(word_info)

    return posts

## :books: Datasets 

This is the inventory of the 17 non-code SE tasks included in SELU and their respective datasets.

The purpose of the notebook inside each folder is exploring the dataset and apply simple transformations such as removing useless columns and normalizing column names to make standard further preprocessing processes. We also make some data quality analysis to identify warnings such as data duplication or class imbalance. At the end of each script, we store the new version of the dataset in parquet or pickle format.

Note that we redefine the *requirement_completion* task to fill only POS verbs; therefore, we also include the corresponding tokenization and POS tagging processes.

|   **Category**  |         **Task**         |                       **Description**                      | **Instances** | **Targets** |
|:---------------:|:------------------------:|:----------------------------------------------------------:|:-------------:|:-----------:|
|    **Binary**   |        _bug_issue_       | Is the reported issue a bug?                               |     38,219    |      2      |
|                 |       _incivility_       | Does the text show unnecessary rude behavior?              |     1,546     |      2      |
|                 |    _requirement_type_    | Is the requirement functional or non-functional?           |      625      |      2      |
|                 |      _tone_bearing_      | Does the text have an unnecessarily disrespectful tone?    |     6,597     |      2      |
| **Multi-class** |     _closed_question_    | Will the question be closed?                               |    140,272    |      5      |
|                 |      _commit_intent_     | Is the commit perfecting or correcting the code?           |     2,533     |      3      |
|                 |       _issue_type_       | Is the issue a bug, an enhancement or a question?          |    803,417    |      3      |
|                 |    _question_quality_    | Has the question a good quality?                           |     60,000    |      3      |
|                 |        _sentiment_       | Which is the sentiment expressed in the text?              |     13,144    |      3      |
| **Multi-label** |    _comment_type_java_   | Which kind of content is detailed in the comment?          |     9,339     |      7      |
|                 |   _comment_type_pharo_   | Which kind of content is detailed in the comment?          |     2,290     |      7      |
|                 |   _comment_type_python_  | Which kind of content is detailed in the comment?          |     1,587     |      5      |
|                 |      _review_aspect_     | Which aspect is involved in the API review?                |     4,522     |      11     |
|                 |        _smell_doc_       | Does the API documentation smell?                          |     1,000     |      5      |
|  **Regression** |      _story_points_      | What is the effort estimated for the task?                 |     23,313    |    [1–96]   |
|     **NER**     |       _se_entities_      | Is it possible to detect specific SE terminology?          |     2,718     |      20     |
|     **MLM**     | _requirement_completion_ | How to fill the specification with the proper user action? |      40*      |      *      |

# NER System for Galician

This repository contains the code necessary to train and run a medical entity recognition system for Galician using the corpus created for the subject Building Language Resources.

The repository offers two ways of training a NER system:
  - using spaCy EntityRecognizer implementation
  - using Huggingface's BERT implementation
  
 To set up the environment, install dependencies using
 
 `pip install -r requirements.txt`
  
 Below, we describe how to use each system.
 
 ### SpaCy
 
 To train the system using spaCy EntityRecognizer, run:
 
 ```
 python main.py spacy \
     --train \                        # if you want to train the system
     --demo \                         # if you want to test the trained system on your data
     --data_dir=<data_dir> \          # the directory containing the training data; only necessary if --train is specified
     --spacy_config=<spacy_config> \  # spaCy configuration file; it is possible to use the one provided in this repository
     --spacy_model=<spacy_model>      # the directory where to store the trained model or where a previously trained model is stored
```
 
For example:

```
python main.py spacy --train --demo --data_dir="./data/galician" --spacy_config="spacy_config.cfg" --spacy_model="spacy_output"
```

### Transformers

To train the system by fine-tuning a pre-trained BERT model, run:

```
python main.py transformer \
     --train \                                  # if you want to train the system
     --demo \                                   # if you want to test the trained system on your data
     --data_dir=<data_dir> \                    # the directory containing the training data; only necessary if --train is specified
     --transformer_model=<transformer_model>    # the directory where to store the trained model or where a previously trained model is stored
```

For example:

```
python main.py transformer --train --demo --data_dir="./data/galician" --transformer_model="transformer_model"
```

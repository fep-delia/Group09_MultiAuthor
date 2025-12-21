MULTI-AUTHOR WRITING STYLE ANALYSIS
This is the repo for Group 9's exam project for the course Advanced Natural Language Processing and Deep Learning at ITU (2025/2026)

The project is based on the PAN (2025) shared task and uses the same training and validation data for the monolingual classification task.
We experiment with various configurations of adapterbased fine-tuning of the multilingual modern BERT model for authorship classification task as well as investigate its multilingual capabilities by evaluating on multilingual data, without training.

Root folder contains all training scripts used for different fine-tuning configurations in the form of Jupyter Notebooks as well as test.ipynb used for evaluation. 
All notebooks should run using the current repo setup, but further instructions can be found within these and within each subdirectory READMEs. Test.ipynb is a notebook used for creating task-predictions for each model configuration.

The Data subdirectory holds the PAN-shared task training data as well as multilingual validation data, sourced from various language specific subreddits, for investigating the multilingual capabilites of the mmBERT model

predictions hold all prediction files made by various model configurations for the task.

Evaluation holds all validation data as well as evaluation scripts

results holds the txt files genereated from the evaluation script in individual subdirectories corresponding the model configuration evaluated.

trained_adapters stores all the adapters trained using the scripts in the root folder for easy implementation and reproducibility. 

Naming conventions used across all folders refer to their source model and adapter configurattion:
e.g. mmbert_lora_lang_2lcls means the adapters being stored/evaluated were trained on the pretrained mmbert model using LoRA with a 2 layer head on top of a previously trained language adapter.


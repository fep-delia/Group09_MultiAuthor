CODE ORIGINALLY SOURCED FROM PAN @ CLEF 2025 (Zangerle et al. 2025)

New adjusted script for multilingual evaluation: evaluator_lang.py

For code to work, first manually create results folders and requires prediction files for a configuration to already have been made using test.py in ROOT.

Evaluation is run in CMD / Anaconda Prompt as:

cd path\to\repo
python Evaluation/evaluator.py -p predictions/model_config_predictions_folder -t Evaluation/validation_data_folder -o results/results_folder

This outputs .txt-file containing F1 scores for each difficulty - also printed in terminal.

EXAMPLE USE FOR MONOLINGUAL EVALUATION:

python Evaluation/evaluator.py -p predictions/predictions_lora_2lcls -t Evaluation/validation_data -o results/results_lora_2lcls

EXAMPLE USE FOR MULTILINGUAL EVALUATION:

python Evaluation/evaluator_lang.py -p predictions/multilingual_predictions_Qlora_language_adapter -t Evaluation/mm_validation -o results/multilingual_results_Qlora_language_adapter

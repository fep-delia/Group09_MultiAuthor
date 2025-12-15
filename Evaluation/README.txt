CODE ORIGINALLY SOURCED FROM PAN @ CLEF 2025 (Zangerle et al. 2025)

Create results folder in base repo
Evaluation run in CMD / Anaconda Prompt

cd path\to\repo
python Evaluation/evaluator.py -p predictions/model_config_predictions_folder -t Evaluation/validation_data_folder -o results/results_folder

This outputs .txt-file containing F1 scores for each difficulty - also printed in terminal.

Example use

python Evaluation/evaluator.py -p predictions/predictions_lora_2lcls -t Evaluation/validation_data -o results/results_lora_2lcls

python Evaluation/evaluator.py -p predictions/predictions_lora_lang -t Evaluation/validation_data -o results/results_lora_lang

python Evaluation/evaluator.py -p predictions/predictions_parallelseqbn -t Evaluation/validation_data -o results/results_parallelseqbn

python Evaluation/evaluator.py -p predictions/predictions_parallelseqbn_2lcls -t Evaluation/validation_data -o results/results_parallelseqbn_2lcls

python Evaluation/evaluator.py -p predictions/predictions_parallelseqbn_lang -t Evaluation/validation_data -o results/results_parallelseqbn_lang

python Evaluation/evaluator.py -p predictions/predictions_parallelseqbn_lang_2lcls -t Evaluation/validation_data -o results/results_parallelseqbn_lang_2lcls

MULTILINGUAL EVALUATION STATEMENTS:

python Evaluation/evaluator_lang.py -p predictions/multilingual_predictions_base -t Evaluation/mm_validation -o results/multilingual_results_base

python Evaluation/evaluator_lang.py -p predictions/multilingual_predictions_lora -t Evaluation/mm_validation -o results/multilingual_results_lora

python Evaluation/evaluator_lang.py -p predictions/multilingual_predictions_lora_2lcls -t Evaluation/mm_validation -o results/multilingual_results_lora_2lcls

python Evaluation/evaluator_lang.py -p predictions/multilingual_predictions_Qlora -t Evaluation/mm_validation -o results/multilingual_results_Qlora

python Evaluation/evaluator_lang.py -p predictions/multilingual_predictions_Qlora_2lcls -t Evaluation/mm_validation -o results/multilingual_results_Qlora_2lcls

python Evaluation/evaluator_lang.py -p predictions/multilingual_predictions_Qlora_language_adapter -t Evaluation/mm_validation -o results/multilingual_results_Qlora_language_adapter

python Evaluation/evaluator_lang.py -p predictions/multilingual_predictions_parallelseqbn -t Evaluation/mm_validation -o results/multilingual_results_parallelseqbn

python Evaluation/evaluator_lang.py -p predictions/multilingual_predictions_parallelseqbn_2lcls -t Evaluation/mm_validation -o results/multilingual_results_parallelseqbn_2lcls

python Evaluation/evaluator_lang.py -p predictions/parallelseqbn_lang_adapter_predictions -t Evaluation/mm_validation -o results/multilingual_results_parallelseqbn_language_adapters

python Evaluation/evaluator_lang.py -p predictions/parallelseqbn_2lcls_lang_adapter_predictions -t Evaluation/mm_validation -o results/multilingual_results_parallelseqbn_2lcls_language_adapters

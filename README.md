# In-Context Learning
This experiment is setup to initially replicate some of the results produced in [1]. The Transformer used is a modification of minGPT from [2].


# Tests
You can view some of the tests run in this folder. Currently you can view the following:
- CombinedBaselines.pdf: A view of the code, run, and output plots
- FunctionDatasetExamples: A view of the code, run using the linear function, and output plots. (Other functions not fully tested)
- TransformerDatasetExamples: A view of the code, run, and output debugging showing how the prompt inputs and targets are fed into the machine
- SimpleTrainingDebug: This is to ensure inputs, targets, and mse update are correct for a small example before moving to a larger example. 
- Training: an initial attempt at training the transformer which plots some of the weights and the training loss.
- Evaluation: an initial attempt at evaluating the trained transformer with varied in-context examples.

# Code
The Code directory contains both the minGPT files and the python files needed for the experiments. 

## The minGPT code
The code for minGPT are the 3 files listed below. The original code is mostly the same. I use a "#MODIFIED" comment to tag the changes I made.
- model.py
- trainer.py
- utils.py
- curriculumtrainer.py (Not yet fully implemented for curriculum training)

## The code used for the experiments are
### Dataset Geeneration
- FunctionDataset.py
- TransformerDataset.py
- CurriculumTransformerDataset.py (Not yet fully implemented for curriculum training)
### Baselines
- CombinedBaselines.py (used to compare all baselines at once)
- LSGDBaseline.py
- LeastSquaresBaseline.py
- AveragingBaseline.py
- KNNBaseline.py
### Transformer Training and Evaluation
- Training.py
- Evaluate.py
- CurriculumTraining.py (Not yet fully implemented for curriculum training)
- CurriculumEvaluate.py (Not yet fully implemented for curriculum training)





# References
[1] Garg, Shivam & Tsipras, Dimitris & Liang, Percy & Valiant, Gregory. (2022). What Can Transformers Learn In-Context? A Case Study of Simple Function Classes.
[2] Andrej Karpathy, minGPT, (2023), GitHub repository,
https://github.com/karpathy/minGPT/blob/master/README.md

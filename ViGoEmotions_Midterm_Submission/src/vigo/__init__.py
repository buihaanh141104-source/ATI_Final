"""ViGoEmotions midterm: multi-label Vietnamese emotion recognition.

Sub-packages
    vigo.data        loading, fixed split, Vietnamese preprocessing, vocabularies, torch loaders
    vigo.models      TF-IDF + LR, BiLSTM, transformer encoder with sigmoid head, losses
    vigo.training    generic training loop, one entry point per model, ensembling
    vigo.evaluation  metrics, threshold strategies, run artefacts (probabilities, summaries)
    vigo.analysis    dataset analysis (EDA), text-style subsets, Midterm questions and error analysis
    vigo.utils       paths, config / JSON I/O, seeding, devices, environment capture
Command line: python -m vigo <command>   (see vigo.cli)
"""

__version__ = "1.1.0"

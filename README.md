Code for the EACL 2021 paper How to Evaluate a Summarizer: Study Design and Statistical Analysis for Manual Linguistic Quality Evaluation

Most experiments can be reproduced by running the accompanying jupyter notebook.
For significance analysis run scrips/r/analyse-ordinal.r anonymized_judgements/<data_file> <score for likert or rank for ranking> crossed

Power analysis is not included in the steps, as it is computationally expensive. To reproduce one step, run 

```bash
python -m summaryanalysis.design_power -b <batch count> -d <docs per batch> -a <annotators per doc> <model_file> out.csv
```


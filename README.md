<!-- Dependencies -->
<!-- install rapidfuzz -->
pip install rapidfuzz


<!-- Commands -->
<!-- test auto-label with 200 samples -->
python scripts/auto_label.py -i data/ai-medical-chatbot.csv -d data --max-sentences 200 --fuzzy-cutoff 85
<!-- auto-label for all sentences -->
python scripts/auto_label.py -i data/ai-medical-chatbot.csv -d data --fuzzy-cutoff 85
<!-- test with 150 generated rows -->
python tests/test_auto_label_sample.py
<!-- test with counts per symptom -->
python scripts/auto_label.py -i data/ai-medical-chatbot.csv -d data -m 2000 --validate -e 3

<!-- Create gold set with 150 stratified entries and a printed summary of what was selected -->
python scripts/select_gold_sample.py

<!-- To test NER -->
<!-- 
    Update model_directory as needed for your file structure.
    Change LABEL_MAP if your label ids are different. 
-->
python scripts/test_infer_ner.py


---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:94763
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: hello doctor,i am trying to conceive but my husband and i did cocaine
    a week ago. how long should my husband and i wait to safely continue to try to
    get pregnant? how long until it is out of our system? how long does cocaine stay
    in sperm? thanks in advance.
  sentences:
  - mouth dryness
  - skin growth
  - ankle swelling
- source_sentence: hi doctor, i am 33 years old. i am coughing from last 1 month.
    in between, it was alright but again started. i do not have any fever, nor chest
    pain. even the sputum which comes out is also normal in color. i do not have any
    problem with sleeping or normal breathing. but, in case when i breathe out from
    the mouth, i get some feeling inside throat to the chest, and i begin to cough.
    currently, i am having cold, and my nasal chambers are all blocked. i have not
    taken any medicine till now. not sure whether i have tb.
  sentences:
  - smoking problems
  - abnormal breathing sounds
  - fever
- source_sentence: hi doctor, i have been suffering from prostate infection for the
    past six months. it started with blood in urine and then burning sensation while
    urinating. i am also having smelly and yellow urine. before this, i got mouth
    ulcers. also, i had itching sensation throughout the body mainly in the chest,
    hand and back. it comes in the evening and night. i had lots of antibiotics, but
    still the issue has not resolved. it seems like sti. i have done the tests for
    hiv, hsv, chlamydia, thpa and i have attached the test results for your reference.
    everything is fine. can it be gonorrhea? as there is no test available for it.
    also, can you please tell me the antibiotics course to be taken for gonorrhea?
    i can try and check if it is resolving the problem.
  sentences:
  - fluid in ear
  - mouth ulcer
  - diarrhea
- source_sentence: hi doctor, i am 50 years old male. i have middle back pain for
    the past eight years. it is usually in the middle right back between the shoulder
    blades. i have this pain when i stand for an extended period. recently, i felt
    the pain radiating from my back into my chest. i think that i have a knot between
    my shoulder blades. it seems to get worse as the day progresses with stress and
    anxiety. after a recent treadmill run, i noticed a knot like a pain in my upper
    right back. it is accompanied by an urge to cough. other new symptoms include
    throat clearing and occasional breathlessness on humid days. but all the pain
    gets relieved with rest and sleep. but it is worse throughout the day. my recent
    blood report showed wbc with 3800 per microliter, rbc with 5.06 million mcl, my
    hemoglobin is 154 g/dl, hematocrit showing 45 %, platelets with 145,000 platelets
    per microliter of blood. my ekg, echocardiogram, stress test and abdominal ultrasound
    were all normal. ultrasound showed a fatty liver. my doctor advised me to do a
    chest x-ray. they found a small lesion on my left lung at the rib one area. he
    said it is nothing and recommended for a ct scan. i am living with full of anxiety.
    i read somewhere that 60 percent of lesions in men are cancer. surely, i am not
    asking for a diagnosis. can a small lesion have such ability to cause so much
    pain and discomfort? is it possible to have advanced lung cancer? but my lungs
    look clear on the rest of the x-ray image. is there anything else in the blood
    work or any other tests showing the possibility of cancer? i have no cough, wheezing,
    fever, loss of appetite, weight loss, and unexplained sweats.
  sentences:
  - eyelid retracted
  - fainting
  - rib pain
- source_sentence: i am presently unable to remail cool, to concentrate effectively
    as lots of fear inside, low self esteem and negative thinking, somw time feel
    restlessness.... i am 34 years old, height 5 feet 9 inch and medical history of
    healthy individual but always feel nervousness.
  sentences:
  - mass or swelling around the anus
  - fever
  - restlessness
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'i am presently unable to remail cool, to concentrate effectively as lots of fear inside, low self esteem and negative thinking, somw time feel restlessness.... i am 34 years old, height 5 feet 9 inch and medical history of healthy individual but always feel nervousness.',
    'restlessness',
    'fever',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9006, 0.4364],
#         [0.9006, 1.0000, 0.3166],
#         [0.4364, 0.3166, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 94,763 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                           | sentence_1                                                                       | label                                                         |
  |:--------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                               | string                                                                           | float                                                         |
  | details | <ul><li>min: 17 tokens</li><li>mean: 128.31 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 4.79 tokens</li><li>max: 11 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.6</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | sentence_1                         | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------|:-----------------|
  | <code>recently tore/cut my frenulum during sex after months of it with no trouble. the frenulum bled heavily but hasn't bled since, which was 4 days ago, i have had sex twice since aswell with some pain. both times i have ejaculated really quickly, do you know why this is?? also the frenulum appears to be twice as thin now with a little infected area containing pus, would sudocrem do the job or do i need to go and see someone??</code>                                                                                                                                                  | <code>fever</code>                 | <code>0.0</code> |
  | <code>i have been through sexual assault.. i lost my virginity at a young age. i have been with 6 diff guys since i was 13. i have depression and anxiety i know that for sure. i am in a relationship with a 17 year old who suffers from psychosis. i believe i ve been going through the same thing.</code>                                                                                                                                                                                                                                                                                          | <code>hip weakness</code>          | <code>0.0</code> |
  | <code>hi, may i answer your health queries right now ? please type your query here...i had surfer with this for two years here the complain goes. the penis is small in size but when i have sex with a girl it will not be more than one minute the penis well be weak and the sperm is camming out in form of water intend of sperm in urine i discover white liquid after pee i consulted on the 08/july/2010 and the following drugs were been given but know change .doxy100mg,nitrociu100mg,flagyl500mg but i see know change. please doctor with great respect and honor i need your help</code> | <code>premature ejaculation</code> | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0844 | 500  | 0.2137        |
| 0.1688 | 1000 | 0.1174        |
| 0.2533 | 1500 | 0.0971        |
| 0.3377 | 2000 | 0.0891        |
| 0.4221 | 2500 | 0.0873        |
| 0.5065 | 3000 | 0.0839        |
| 0.5909 | 3500 | 0.0805        |
| 0.6753 | 4000 | 0.0784        |
| 0.7598 | 4500 | 0.0787        |
| 0.8442 | 5000 | 0.0768        |
| 0.9286 | 5500 | 0.0751        |


### Framework Versions
- Python: 3.12.12
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.8.0+cu126
- Accelerate: 1.11.0
- Datasets: 4.0.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
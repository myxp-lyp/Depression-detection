## Depression-detection
### Data preprocessing section

[audio_preprocess.py](https://github.com/myxp-lyp/Depression-detection/tree/main/Data%20preprocess/): This file is used to complete the entire audio preprocess process, starting from the zip files downloaded, to the reorganized audios that can be directly used by the models. This file can be run directly.

[cut_transcript.py](https://github.com/myxp-lyp/Depression-detection/tree/main/Data%20preprocess/cut_transcript.py) and [merge_transcript.py](https://github.com/myxp-lyp/Depression-detection/tree/main/Data%20preprocess/merge_transcript.py): These two files are used to preprocess the transcript, the former one should be run before the latter one. 

[crawl_data.ipynb](https://github.com/myxp-lyp/Depression-detection/tree/main/Data%20preprocess/crawl_data.ipynb): This file is used to crawl the passages from WHO. This file should be run from head to tail in a windows computer (or perhaps linux with chrome core correctly set up). This file can be run directly.

[q&a_preprocess.py](https://github.com/myxp-lyp/Depression-detection/tree/main/Data%20preprocess/q&a_preprocess.py): This file is used to generate the q&a set. You need to fill in the API_KEY section before you run this, if you don't have one, you would have to sign up an DeepSeek account at [here](https://platform.deepseek.com/sign_up), or use third party services (then you will also need to modify DEEPSEEK_API_URL). **This will cost approximately 1.55 CNY (0.21 USD).**

---

### Model training section

*It is recommended to completely run all files under data preprocessing section before running files below.*

> There are four models used in this task, named as "transcript", "qa_transcript",  "audio" and "qa_audio". The meaning of the labels are shown below:
>
> qa: The base model was fine-tuned on the Q&A dataset created, otherwise, the base model was the original Llama model.
>
> transcript: The model was trained on the processed transcript file.
>
> audio: The model was trained on the processed audio file.

[qa.py](https://github.com/myxp-lyp/Depression-detection/blob/main/Model%20training/qa_train.py): This file uses the Q&A dataset gained in the data preprocessing section, and fine-tune the Llama model, generating lora checkpoints and model that can be used for other models. This file can be run directly. 

[transcript.py](https://github.com/myxp-lyp/Depression-detection/blob/main/Model%20training/transcript.py), [qa_transcript.py](https://github.com/myxp-lyp/Depression-detection/blob/main/Model%20training/qa_transcript.py): These two files are used to train and evaluate the "transcript" and "qa_transcript" model respectively. This file can be run directly.
> Note that in order to use the model created by this file, you will need to add the following code:
'''
from safetensors.torch import load_file, save_file

sd = load_file("/path/to/bad_adapter.safetensors")

new_sd = {}
for k, v in sd.items():
    if k.startswith("base_model.model."):
        new_key = k.replace("base_model.model.", "", 1)
    else:
        new_key = k
    new_sd[new_key] = v

save_file(new_sd, "/path/to/fixed_adapter.safetensors")
'''

[audio.py](https://github.com/myxp-lyp/Depression-detection/blob/main/Model%20training/audio.py): This file is used to train both "audio" and "qa_audio" model. To train these model, you will need to determine the base model used between line 197 and line 228. 

[evaluate.py](https://github.com/myxp-lyp/Depression-detection/blob/main/Model%20training/evaluate.py): This file is used to evaluate the performance of the fine-tuned Llama model(qa). This file uses PsycholexLlama to ask both fine-tuned and original model, and score their answer. Before running this file, you would have to log in to your Hugging face account and apply for the usage of PsycholexLlama. 

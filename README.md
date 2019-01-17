### Contextual Slot Filling for Task-Oriented Conversational Agents

Memory-enhanced CRF implementation for the slot filling problem for conversational agents. Uses dialogue acts in explicit feature extraction step. Requires the input to be in the same style as the [simulated dialogue](https://github.com/google-research-datasets/simulated-dialogue) dataset by Google. 

This is the code I used for my bachelor thesis ([Contextual Slot Filling for Task-Oriented Conversational Agents](http://scriptiesonline.uba.uva.nl/en/scriptie/659240)). 

[Original ME-CRF](https://github.com/liufly/mecrf) code by Liu et al. 

Prerequisites:
```
Python 2.7
TensorFlow 1.3
numpy
Flask
```

Training:
```shell
$ python csf.py --embedding_file PATH/TO/WORD_EMBEDDING_FILE
```
See the csf.py file for all other possible flags.
Evaluating:
```shell
$ python predict.py
```
See the predict.py file for the possible flags.

Run as Flask server to accept new inputs as body of HTTP POST requests:
```shell
$ FLASK_APP=server.py flask run
```

Structure the input the same as the simulated dialogue dataset:
```
[
    {
        "turns": [
            {
                "user_utterance": {
                    "text": "hello , can you find me a restaurant ?",
                    "slots": []
                },
                "user_acts": [
                    {
                        "type": "GREETING"
                    }
                ]
            },
            {
                "system_utterance": {
                    "text": "how much would you like to spend ?",
                    "slots": []
                },
                "system_acts": [
                    {
                        "slot": "price_range",
                        "type": "REQUEST"
                    }
                ],
                "user_utterance": {
                    "text": "something that is inexpensive .",
                    "slots": [
                        {
                            "slot": "price_range",
                            "start": 3,
                            "exclusive_end": 4
                        }
                    ]
                },
                "user_acts": [
                    {
                        "type": "INFORM"
                    }
                ]
            }
        ]
    }
]

```

Original paper:  
##### Capturing Long-range Contextual Dependencies with Memory-enhanced Conditional Random Fields (IJCNLP-2017)
```
@InProceedings{Liu+:2017,
  author    = {Liu, Fei  and  Baldwin, Timothy  and  Cohn, Trevor},
  title     = {Capturing Long-range Contextual Dependencies with
Memory-enhanced Conditional Random Fields},
  booktitle = {Proceedings of the Eighth International Joint Conference on Natural Language Processing (IJCNLP 2017)},
  year      = {2017},
  address   = {Taipei, Taiwan},
  pages     = {555--565}
}
```
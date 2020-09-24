# Recognition of emotions based on analysis of audio data using convolucional neural networks

The objective of this research project is to assess the feeling of a client during a conversation with a chabot or human agent who recognizes a voice in real time. Analyzing the customer's feeling in various parts of the call can help to understand the transition of the customer's emotion and direct the bot/human to give the correct answers.

## Motivation

The ability to communicate between individuals through speech is essential. Speech serves to convey a message through words. However, a person's speech can be altered by several changes in the nervous system. For example, the speech produced by a person in a state of fear, anger or joy becomes loud and fast, with an increasingly high range; while emotions such as sadness or tiredness generate slow and low speech.

Through the analysis of these characteristics by Artificial Intelligence (AI) algorithms, which tries to solve problems with intuitive solutions, it becomes possible to automatically recognize the emotion of the interlocutor. Vocal parameters and prosodic characteristics, such as pitch and speech rate variables, can be analyzed using computational pattern recognition techniques. That is, some emotions can be identified, such as joy or sadness, for example.

The main reason for the development of this research project is the fact that there is still no system for recognizing emotions that works with acceptable precision and in a generic way. A system with such precision, it can be quite useful to recognize is the possibility of using the recognition of emotions to improve the interaction between humans and machines. In this context, if a robot knows the emotion of the human with which it is interacting, it can give an answer that best suits the moment or situation, and not just another standard answer - as is very common today. With this, the use of Convolutional Neural Networks aims to supply, in parts, this need. 

## Dataset

* RAVDESS: This dataset includes around 1500 audio file input from 24 different actors. 12 male and 12 female where these actors record short audios in 8 different emotions i.e 1 = neutral, 2 = calm, 3 = happy, 4 = sad, 5 = angry, 6 = fearful, 7 = disgust, 8 = surprised. It can be found [here](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) or [here](https://smartlaboratory.org/ravdess/). Dataset consists of different emotions like -  *neutral*, *calm*, *happy*, *sad*, *angry*, *fearful*, *disgust*, *surprised*.

### Team Members

Leonardo Barros, Iftekhar Ahmed, Eduardo Almeida, Ian Garnerh


## Process

We approach this problem in three stages:

**Stage 1** - We perform source separation on the audio conversation, by performing VAD detection on the conversation and dividing the audio conversation into different chunks, on each chunk we apply GMM and a global GMM (UBM) on the whole conversation, using BIC and through spectral clustering we cluster every chunk into different speakers.

**Stage 2** – We then apply sentiment analysis on supervised speech emotion dataset (RAVDESS) using Deep Neural Networks.

**Stage 3** - We used the trained model from Stage 2 to classify the sentiment of the speaker chunks.

## File description

`convert.py` helps convert *.mp3* audio files into *.wav*

`extract_data.py` is used to extract files from the RAVDESS dataset and store it together on the basis of emotions

`speaker_diarization.py` is used to separate audio chunks of customer and chatbot or human chat platform. Once the chunks are separated, only chunks containing customer's voice are considered for sentiment analysis.

`sentiment_classification.py` contains complete code on how various audio features that are extracted to train and test the model along with different architectures created to carry out the experiments. We are using Keras to do this.
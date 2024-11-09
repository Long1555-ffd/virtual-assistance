This brand is for making plan and debuging

All the things we should do ahead:

- the innoai/Edge-Text-TO-Speech model is not available on huggingface, so we can think about other options

- Doing some research about running quantization and offloading techniques 
(Running the model on our local computer is easy, but running it in a more optimized and efficient way is a different story)

- We also need to fine-tune the LLM models, so that it can be adaptable and versatile for our own usage

- We also need to combine TTS and LLM model, using asycn/await method to run these models asychronously

- Testing with the speech recognition also

- Here is the workflow of the model that I am trying to make it: speech input -> speech recognition (converts from speech to text) -> LLM for natural language processing -> the text output will be sent into the TTS model -> output the speech back on the users' headphones/headsets.

- For more advanced workflow like some models (chatgpt o1 - advanced voice mode), the input is speech and the output is speech directly, without converting speech-text or text-speech back and forth
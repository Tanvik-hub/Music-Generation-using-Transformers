# 🎵Music-Generation-using-Transformers

The **Music Transformer**, or Transformer Decoder with Relative Self-Attention, is a deep learning sequence model designed to generate music. It builds upon the Transformer architecture to consider the relative distances between different elements of the sequence, rather than/along with their absolute positions in the sequence. I explored my interest in AI-generated music through this project and learned quite a bit about current research in the field of AI in terms of both algorithms and architectures. This repository contains Python scripts to preprocess MIDI data, train a pre-LayerNorm Music Transformer using PyTorch, and generate MIDI files with a trained (or if you're brave, untrained) Music Transformer. It also contains three of my own trained Music Transformers that can be used to generate music.

While the data preprocessing and generation functionality require MIDI files and the event vocabulary described in Oore et al. 2018 (or `vocabulary.py`), anyone should be able to use the `train.py` script to train their own Relative Attention Transformer on any dataset, provided correct specification of hyperparameters, and provided they have properly preprocessed their data into a single PyTorch tensor. Do create an issue if something does not work as expected.

Follow the instructions in the `Generate_Music.ipynb` notebook, or in the **Generate Music!** section of this README to generate your own music with one of my pretrained Music Transformers, possibly without having to write any code.

## Key Dependencies
- PyTorch 2.1.0
- Mido 1.2.9

## Setting up
Clone the git repository, navigate into it, and install the requirements. Then you're ready to preprocess MIDI files, train, and generate music with a Music Transformer.

```bash
git clone https://github.com/spectraldoy/music-transformer
cd ./music-transformer
pip install -r requirements.txt
```

# Generate Music!
The Music Transformer is useless if we can't generate music with it. Given a pretrained Music Transformer’s state_dict and hparams saved at .../save_path.pt, and specifying the path to save a generated MIDI file, .../gen_audio.mid, run the following:
```bash
python generate.py .../save_path.pt .../gen_audio.mid
```
This will autoregressively greedy decode the outputs of the Music Transformer to generate a list of token_ids, convert those token_ids back to a MIDI file using functionality from tokenizer.py, and will save the output MIDI file at .../gen_audio.mid. Parameters for the MIDI generation can also be specified, such as argmax or categorical decode sampling, sampling temperature, the number of top_k samples to consider, and the approximate tempo of the generated audio (for more details, run python generate.py -h).

I have uploaded a few small pretrained Music Transformers to this repository, with which anyone can run this script to generate music. The models are:

->model4v2: Contains absolute positional encoding up to 20,000 positions and otherwise the exact hparams of hparams.py. It was trained on about 100 MIDI files from the MAESTRO Dataset.
->model6v2: A pure Relative Attention model, containing no absolute positional encoding, following the exact hparams of hparams.py, and trained on the same set of MIDI files as model4v2.
->chopintransformerv5: A pure relative attention model, trained on a set of 39 pieces by Chopin. However, it sounds nothing like him. This is arguably my best model.
->vgmtransformerv4: Trained on the Video Game Music folder of the ADL Piano MIDI Dataset.

To generate music with the chopintransformerv5, you can run:
```bash
python generate.py models/chopintransformerv5.pt .../gen_audio.mid -v
```

I have found that a sampling temperature of 0.7-1.0 and top_k of 50-200 work well with this model. Sometimes, however, it doesn’t end.

The notebook Generate_Music.ipynb allows you to generate music with the models in this repository (by default the Chopin model) using Google’s Magenta SoundFont, as well as download any generated audio files, without having to write any underlying code. So for those who wish to play around with these models, go ahead and open that notebook in Google Colab.


# Preprocess MIDI Data
Most sequence models require a general upper limit on the length of the sequences being modeled, as it becomes computationally or memory expensive to handle longer sequences. Suppose you have a directory of MIDI files at .../datapath/ (for instance, any of the folders in the MAESTRO Dataset), and would like to convert these files into an event vocabulary that can be trained on, cut these sequences to be less than or equal to an approximate maximum length, lth, and store this processed data in a single PyTorch tensor for use with torch.utils.data.TensorDataset at .../processed_data.pt. Running the preprocessing.py script as follows:
```bash
python preprocessing.py .../datapath/ .../processed_data.pt lth

```
This will translate the MIDI files to the event vocabulary laid out in vocabulary.py, tokenize it with functionality from tokenizer.py, cut the data to approximately the specified lth, augment the dataset by a default set of pitch transpositions and time stretches, and finally, store the sequences as a single concatenated PyTorch tensor at .../processed_data.pt. Pitch transpositions and time stretch factors can also be specified when running the script (for details, run python preprocessing.py -h).

Note: This script will not work properly for multi-track MIDI files, and any other instruments will automatically be converted to piano, as I worked only with single-track piano MIDI for this project.


# Train a Music Transformer🎵
The Music Transformer is highly space-complex and requires significant time to train on both GPUs and TPUs, so checkpointing while training is essential. The MusicTransformerTrainer class in train.py implements checkpointing. At its simplest, given a path to a preprocessed dataset in the form of a PyTorch tensor, .../preprocessed_data.pt, a path to checkpoint the model, .../ckpt_path.pt, a path to save the model, .../save_path.pt, and the number of epochs to train the model, running the following:

```bash
python train.py .../preprocessed_data.pt .../ckpt_path.pt .../save_path.pt epochs
```


This splits the data 80/20 into training and validation sets, trains the model for the specified number of epochs, prints progress messages, and checkpoints the optimizer state, learning rate schedule state, model weights, and hyperparameters on encountering a KeyboardInterrupt, anytime a progress message is printed, and when the model finishes training for the specified number of epochs. Hyperparameters can also be specified when creating a new model (for details, run python train.py -h).

If the -l or --load-checkpoint flag is specified, you can resume training from a checkpoint as follows:


```bash
python train.py .../preprocessed_data.pt .../ckpt_path.pt .../save_path.pt epochs -l
```

The latest checkpoint stored at .../ckpt_path.pt will be loaded, and training will continue. Once training is complete, another checkpoint will be created, and the model’s state_dict and hparams will be saved in a Python dictionary at .../save_path.pt.

# Acknowledgements
I trained most of my models on Western classical music from the MAESTRO Dataset.
The model trained on video game music was trained using a subset of the ADL Piano MIDI Dataset.


# you can use the above colab link and generate piano music






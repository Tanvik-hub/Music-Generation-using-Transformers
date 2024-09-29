# Music-Generation-using-Transformers

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

#Generate Music!
The Music Transformer is useless if we can't generate music with it. Given a pretrained Music Transformer’s state_dict and hparams saved at .../save_path.pt, and specifying the path to save a generated MIDI file, .../gen_audio.mid, run the following:

python generate.py .../save_path.pt .../gen_audio.mid




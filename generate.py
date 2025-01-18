

import torch
import mido
import time
import argparse
from masking import *
from tokenizer import *
from vocabulary import *



def load_model(filepath, compile = False):
   
    from model import MusicTransformer
    from hparams import hparams, device
    
    file = torch.load(filepath, map_location=device)
    if "hparams" not in file:
        file["hparams"] = hparams

    model = MusicTransformer(**file["hparams"]).to(device)
    model.load_state_dict(file["state_dict"])

    if compile:
        model = torch.compile(model)

    model.eval()
    return model


def greedy_decode(model, inp, mode="categorical", temperature=1.0, k=None):
    """
    The transformer is an autoregressive model, which means that at the inference stage, it makes next predictions
    based on its previous outputs. This method implements the decoding functionality for autoregressive computation
    of outputs. This function will let the model generate until it predicts an end token, or encounters a
    RuntimeError or KeyboardInterrupt. The greedy in greedy_decode comes from the fact that we are not considering
    the entire output of the model at each step as the true output, and are simply iteratively clipping the very last
    index of the output and appending it to the input to the model, to autoregressively compute the next output.

    Args:
        model: MusicTransformer model whose outputs to greedily decode
        inp (list): list of midi events in the vocabulary for the model to continue; set simply to ["<start>"] for the
                    model to generate from scratch
        mode (str): specify 'categorical' or 'argmax' decode sampling; default: categorical
        temperature (float ~ 1.0): softmax temperature to make the model outputs more diverse (high temperature) or less
                                   diverse (low temperature); default: 1.0
        k (int): number of top k samples to categorically sample to get the predicted next token; default: None, i.e.,
                 all samples will be considered, not just the top k

    Returns:
        torch.LongTensor of token_ids autoregressively generated by the model
    """
    # convert input tokens to list of token ids
    inp = events_to_indices(inp)

    # make sure inp starts with the start token
    if inp[0] != start_token:
        inp = [start_token] + inp

    # convert to torch tensor and convert to correct dimensions for masking
    inp = torch.tensor(inp, dtype=torch.int64, device=device)
    inp = inp.unsqueeze(0)
    n = inp.dim() + 2

    # parameters for decode sampling
    if not callable(temperature):
        temperature__ = temperature
        del temperature

        def temperature(x):
            return temperature__

    if k is not None and not callable(k):
        k__ = k
        del k

        def k(x):
            return k__

    # autoregressively generate output
    torch.set_float32_matmul_precision("high")
    try:
        with torch.no_grad():
            while True:
                # get next predicted idx
                predictions = model(inp, mask=create_mask(inp, n))
                # divide logits by temperature as a function of current length of sequence
                predictions /= temperature(inp[-1].shape[-1])

                # sample the next predicted idx
                if mode == "argmax":
                    prediction = torch.argmax(predictions[..., -1, :], dim=-1)

                elif k is not None:
                    # get top k predictions, where k is a function of current length of sequence
                    top_k_preds = torch.topk(predictions[..., -1, :], k(inp[-1].shape[-1]), dim=-1)
                    # sample top k predictions
                    predicted_idx = torch.distributions.Categorical(logits=top_k_preds.values[..., -1, :]).sample()
                    # get the predicted id
                    prediction = top_k_preds.indices[..., predicted_idx]

                elif mode == "categorical":
                    prediction = torch.distributions.Categorical(logits=predictions[..., -1, :]).sample()

                else:
                    raise ValueError("Invalid mode or top k passed in")

                # if we reached the end token, immediately output
                if prediction == end_token:
                    return inp.squeeze()

                # else cat and move to the next prediction
                inp = torch.cat(
                    [inp, prediction.view(1, 1)],
                    dim=-1
                )

    except (KeyboardInterrupt, RuntimeError):
        # generation takes a long time, interrupt in between to save whatever has been generated until now
        # RuntimeError is in case the model generates more tokens that there are absolute positional encodings for
        pass

    # extra batch dimension needs to be gotten rid of, so squeeze
    return inp.squeeze()


def audiate(token_ids, save_path="gneurshk.mid", tempo=512820, verbose=False):
    """
    Turn a list of token_ids generated by the model (or simply translated from a midi file to the event vocabulary)
    into a midi file and save it. It was also planned to convert the generated file into .wav and/or .flac files,
    however the overhead of getting it to work with ambiguous paths of soundfonts is tedious, and so the conversion
    of generated midi files to .wav or .flac files has been (for now at least) left to the user.

    Args:
        token_ids (torch.Tensor): one-dimensional tensor of token_ids generated by a MusicTransformer
        save_path (str): path at which to save the midi file
        tempo (int): approximate tempo in µs / beat of the generated midi file; default: 512820 (the tempo of all midi
                     files in the MAESTRO dataset
        verbose (bool): flag for verbose output; default: False

    Returns:
        Nothing, but saves the generated tokens in a midi file at save_path
    """
    # set file to a midi file
    if save_path.endswith(".midi"):
        save_path = save_path[:-1]
    elif save_path.endswith(".mid"):
        pass
    else:
        save_path += ".mid"

    # create and save the midi file
    print(f"Saving midi file at {save_path}...") if verbose else None
    mid = list_parser(index_list=token_ids, fname=save_path[:-4], tempo=tempo)
    mid.save(save_path)

    """ save other file formats
    if save_flac:
        flac_path = save_path[:-4] + ".flac"
        print(f"Saving flac file at {flac_path}...") if verbose else None
        fs = FluidSynth()
        fs.midi_to_audio(save_path, flac_path)

    if save_wav:
        wav_path = save_path[:-4] + ".wav"
        print(f"Saving wav file at {wav_path}...") if verbose else None
        fs = FluidSynth()
        fs.midi_to_audio(save_path, wav_path)

        # useful for ipynbs
        return Audio(wav_path)
    """
    
    print("Done")
    return


def generate(model_, inp, save_path="./bloop.mid", mode="categorical", temperature=1.0, k=None,
             tempo=512820, verbose=False):
    """
    Combine the above 2 functions into a single function for easy generation of audio files with a MusicTransformer,
    i.e. greedy_decode followed by audiate

    NOTE: this can take a long time, even on a GPU

    Args:
        model_: MusicTransformer model with which to generate audio
        inp (list): list of midi events in the vocabulary for the model to continue; set simply to ["<start>"] for the
                    model to generate from scratch
        save_path (str): path at which to save the midi file
        mode (str): specify 'categorical' or 'argmax' decode sampling; default: categorical
        temperature (float ~ 1.0): softmax temperature to make the model outputs more diverse (high temperature) or less
                                   diverse (low temperature); default: 1.0
        k (int): number of top k samples to categorically sample to get the predicted next token; default: None, i.e.,
                 all samples will be considered, not just the top k
        tempo (int): approximate tempo in µs / beat of the generated midi file; default: 512820 (the tempo of all midi
                     files in the MAESTRO dataset
        verbose (bool): flag for verbose output; default: False

    Returns:
        Nothing, but saves the audio generated by the model_ as a midi file at the specified save_path
    """
    # greedy decode
    print("Greedy decoding...") if verbose else None
    start = time.time()
    token_ids = greedy_decode(model=model_, inp=inp, mode=mode, temperature=temperature, k=k)
    end = time.time()
    print(f"Generated {len(token_ids)} tokens.", end=" ") if verbose else None
    print(f"Time taken: {round(end - start, 2)} secs.") if verbose else None

    # generate audio
    return audiate(token_ids=token_ids, save_path=save_path, tempo=tempo, verbose=verbose)


if __name__ == "__main__":
    from hparams import hparams

    def check_positive(x):
        if x is None:
            return x
        x = int(x)
        if x <= 0:
            raise argparse.ArgumentTypeError(f"{x} is not a positive integer")
        return x

    parser = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate midi audio with a Music Transformer!"
    )
    parser.add_argument("path_to_model", help="string path to a .pt file at which has been saved a dictionary "
                                              "containing the model state dict and hyperparameters", type=str)
    parser.add_argument("save_path", help="path at which to save the generated midi file", type=str)
    
    parser.add_argument("-c", "--compile", help="if true, model will be `torch.compile`d for potentially better "
                                                "speed; default: false", action="store_true")
    parser.add_argument("-m", "--mode", help="specify 'categorical' or 'argmax' mode of decode sampling", type=str)
    parser.add_argument("-k", "--top-k", help="top k samples to consider while decode sampling; default: all",
                        type=check_positive)
    parser.add_argument("-t", "--temperature",
                        help="temperature for decode sampling; lower temperature, more sure the sampling, "
                             "higher temperature, more diverse the output; default: 1.0 (categorical sample of true "
                             "model output)",
                        type=float)
    parser.add_argument("-tm", "--tempo", help="approximate tempo of generated sample in BMP", type=check_positive)
    parser.add_argument("-i", "--midi-prompt", help="if specified, the program will "
                        "generate music that continues the input midi file", type=str)
    parser.add_argument("-it", "--midi-prompt-tokens", help="number of tokens to sample "
                        "from the midi-prompt input as a prefix to continue, if it has been specified", type=int)
    parser.add_argument("-v", "--verbose", help="verbose output flag", action="store_true")

    args = parser.parse_args()

    # fix arguments
    temperature_ = float(args.temperature) if args.temperature else 1.0
    mode_ = args.mode if args.mode else "categorical"
    k_ = int(args.top_k) if args.top_k else None
    tempo_ = int(60 * 1e6 / int(args.tempo)) if args.tempo else 512820

    if args.midi_prompt:
        midi_parser_output = midi_parser(args.midi_prompt)
        tempo_ = midi_parser_output[2]
        midi_input = (midi_parser_output[1])[0:args.midi_prompt_tokens] if args.midi_prompt_tokens else midi_parser_output[1]
    else:
      midi_input = ["<start>"]
    
    music_transformer = load_model(args.path_to_model, args.compile)
    generate(model_=music_transformer, inp=midi_input, save_path=args.save_path,
             temperature=temperature_, mode=mode_, k=k_, tempo=tempo_, verbose=args.verbose)

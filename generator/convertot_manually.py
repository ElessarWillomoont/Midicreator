def tokens_to_midi(
    self,
    tokens: TokSequence | list[TokSequence] | list[int | list[int]] | np.ndarray,
    programs: list[tuple[int, bool]] | None = None,
    output_path: str | None = None,
) -> Score:
    r"""
    Detokenize one or multiple sequences of tokens into a MIDI file.

    You can give the tokens sequences either as :class:`miditok.TokSequence`
    objects, lists of integers, numpy arrays or PyTorch/Jax/Tensorflow tensors.
    The MIDI's time division will be the same as the tokenizer's:
    ``tokenizer.time_division``.

    :param tokens: tokens to convert. Can be either a list of
        :class:`miditok.TokSequence`, a Tensor (PyTorch and Tensorflow are
        supported), a numpy array or a Python list of ints. The first dimension
        represents tracks, unless the tokenizer handle tracks altogether as a
        single token sequence (``tokenizer.one_token_stream == True``).
    :param programs: programs of the tracks. If none is given, will default to
        piano, program 0. (default: ``None``)
    :param output_path: path to save the file. (default: ``None``)
    :return: the midi object (``symusic.Score``).
    """
    if not isinstance(tokens, (TokSequence, list)) or (
        isinstance(tokens, list)
        and any(not isinstance(seq, TokSequence) for seq in tokens)
    ):
        tokens = self._convert_sequence_to_tokseq(tokens)

    # Preprocess TokSequence(s)
    if isinstance(tokens, TokSequence):
        self._preprocess_tokseq_before_decoding(tokens)
    else:  # list[TokSequence]
        for seq in tokens:
            self._preprocess_tokseq_before_decoding(seq)

    midi = self._tokens_to_midi(tokens, programs)

    # Create controls for pedals
    # This is required so that they are saved when the MIDI is dumped, as symusic
    # will only write the control messages.
    if self.config.use_sustain_pedals:
        for track in midi.tracks:
            for pedal in track.pedals:
                track.controls.append(ControlChange(pedal.time, 64, 127))
                track.controls.append(ControlChange(pedal.end, 64, 0))
            if len(track.pedals) > 0:
                track.controls.sort()

    # Set default tempo and time signatures at tick 0 if not present
    if len(midi.tempos) == 0 or midi.tempos[0].time != 0:
        midi.tempos.insert(0, Tempo(0, self.default_tempo))
    if len(midi.time_signatures) == 0 or midi.time_signatures[0].time != 0:
        midi.time_signatures.insert(0, TimeSignature(0, *TIME_SIGNATURE))

    # Write MIDI file
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        midi.dump_midi(output_path)
    return midi
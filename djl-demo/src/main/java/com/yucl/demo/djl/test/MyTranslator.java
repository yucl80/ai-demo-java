package com.yucl.demo.djl.test;

import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

/** The {@link ai.djl.translate.Translator} for PyTorch GPT2 model. */
public class MyTranslator implements NoBatchifyTranslator<NDList, CausalLMOutput> {

    private long kvDim;
    private int numAttentionHeads;
    private int numLayers;

    public MyTranslator() {

    }

    /**
     * Constructs a new instance of {@code PtGptTranslator}.
     *
     * @param kvDim             the kv dimension
     * @param numAttentionHeads the number of attention heads
     * @param numLayers         the number of layers
     */
    public MyTranslator(long kvDim, int numAttentionHeads, int numLayers) {
        this.kvDim = kvDim;
        this.numAttentionHeads = numAttentionHeads;
        this.numLayers = numLayers;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, NDList input) throws Exception {
        // input = [inputIds, posIds, attnMask]
        NDManager manager = ctx.getNDManager();
        NDArray inputIds = input.get(0);
        inputIds.setName("input_ids");

        NDArray attentionMask = input.get(2);
        attentionMask.setName("attention_mask");

        NDList inputNew;
        if (input.size() == 3) {
            // pastKeyValue == null
            NDArray useCacheBranch = manager.create(new boolean[] { false }, new Shape(1));
            useCacheBranch.setName("position_ids");
            inputNew = new NDList(inputIds, attentionMask, useCacheBranch);
            initialDummyPastKeyValues(inputIds, manager, inputNew);
        } else {
            NDArray useCacheBranch = manager.create(new boolean[] { true }, new Shape(1));
            useCacheBranch.setName("use_cache_branch");
            inputNew = new NDList(inputIds, attentionMask, useCacheBranch);
            inputNew.addAll(input.subNDList(3));
        }

        int offset = 3;
        // for (int i = offset; i < numLayers * 2 + offset; i += 2) {
        for (int i = offset; i < 22; i += 2) {
            int order = (i - offset) / 2;
            inputNew.get(i).setName(String.format("past_key_values.%s.key", order));
            inputNew.get(i + 1).setName(String.format("past_key_values.%s.value", order));
        }

        return inputNew;
    }

    /** {@inheritDoc} */
    @Override
    public CausalLMOutput processOutput(TranslatorContext ctx, NDList output) throws Exception {
        return new CausalLMOutput(output.get(0), output.subNDList(1));
    }

    private void initialDummyPastKeyValues(NDArray inputIds, NDManager manager, NDList list) {
        long numBatch = inputIds.getShape().get(0);
        for (int i = 0; i < 23; ++i) {
            NDArray array = manager.zeros(new Shape(2, 2, 32, 64));
            list.add(array);
        }
    }
}

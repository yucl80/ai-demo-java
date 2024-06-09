package ai.djl.modality.nlp.generate;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;
import java.util.Arrays;
import java.util.Collection;
import java.util.function.Function;
import java.util.stream.Collectors;

public class MyTextGenerator {
    private String searchName;
    private SearchConfig config;
    private Predictor<NDList, CausalLMOutput> predictor;
    private NDArray positionOffset;
    private long[] endPosition;

    public MyTextGenerator(Predictor<NDList, CausalLMOutput> predictor, String searchName, SearchConfig searchConfig) {
        this.predictor = predictor;
        this.searchName = searchName;
        this.config = searchConfig;
    }

    public NDArray greedySearch(NDArray inputIds) throws TranslateException {
        this.endPosition = new long[Math.toIntExact(inputIds.getShape().get(0))];
        Arrays.fill(this.endPosition, (long) this.config.getMaxSeqLength());
        NDArray attentionMask = this.prepareAttentionMaskOffset(inputIds, this.config);
        NDManager manager = inputIds.getManager();
        GreedyBatchTensorList searchState = new GreedyBatchTensorList(inputIds, (NDArray) null, (NDList) null,
                attentionMask);

        do {
            NDScope ignore = new NDScope();

            long tokenId;
            try {
                NDArray pastOutputIds = searchState.getPastOutputIds();
                NDArray nextInputIds = searchState.getNextInputIds();
                NDArray pastAttentionMask = searchState.getPastAttentionMask();
                NDList pastKeyValues = searchState.getPastKeyValues();
                tokenId = pastOutputIds == null ? 0L : pastOutputIds.getShape().getLastDimension();
                NDList modelInput = this.prepareInput(nextInputIds, pastAttentionMask, tokenId, 1);
                if (pastKeyValues != null) {
                    modelInput.addAll(pastKeyValues);
                }

                CausalLMOutput modelOutput = (CausalLMOutput) this.predictor.predict(modelInput);
                NDArray outputIds = StepGeneration.greedyStepGen(modelOutput.getLogits());
                if (pastOutputIds == null) {
                    pastOutputIds = nextInputIds;
                    searchState.setPastOutputIds(nextInputIds);
                } else {
                    pastOutputIds = pastOutputIds.concat(nextInputIds, 1);
                    searchState.setPastOutputIds(pastOutputIds);
                }

                searchState.setNextInputIds(outputIds);
                pastKeyValues = modelOutput.getPastKeyValuesList();
                searchState.setPastKeyValues(pastKeyValues);
                pastAttentionMask = pastAttentionMask.concat(
                        manager.ones(new Shape(new long[] { inputIds.getShape().get(0), 1L }), DataType.INT64), 1);
                searchState.setPastAttentionMask(pastAttentionMask);
                NDScope.unregister(new NDArray[] { outputIds, pastAttentionMask, pastOutputIds });
                NDScope.unregister(pastKeyValues);
            } catch (Throwable var16) {
                try {
                    ignore.close();
                } catch (Throwable var15) {
                    var16.addSuppressed(var15);
                }

                throw var16;
            }

            ignore.close();
            long[] outputIdsArray = searchState.getNextInputIds().toLongArray();

            for (int i = 0; i < this.endPosition.length; ++i) {
                long[] var19 = outputIdsArray;
                int var20 = outputIdsArray.length;

                for (int var21 = 0; var21 < var20; ++var21) {
                    tokenId = var19[var21];
                    if (tokenId == this.config.getEosTokenId()) {
                        this.endPosition[i] = searchState.getPastOutputIds().getShape().get(1) + 1L;
                        break;
                    }
                }
            }
        } while (searchState.getPastOutputIds().getShape().get(1) + 1L < (long) this.config.getMaxSeqLength());

        return searchState.getPastOutputIds().concat(searchState.getNextInputIds(), 1);
    }

    public NDArray beamSearch(NDArray inputIds) throws TranslateException {
        this.endPosition = new long[Math.toIntExact(inputIds.getShape().get(0))];
        Arrays.fill(this.endPosition, (long) this.config.getMaxSeqLength());
        NDArray attentionMask = this.prepareAttentionMaskOffset(inputIds, this.config);
        NDManager manager = inputIds.getManager();
        long numBeam = (long) this.config.getBeam();
        long numBatch = inputIds.getShape().get(0);
        BeamBatchTensorList searchState = new BeamBatchTensorList();
        long numHeads = 0L;
        long kvDim = 0L;

        do {

            if (searchState.getPastAttentionMask() == null) {
                NDList modelInput = this.prepareInput(inputIds, attentionMask, 0L, 1);
                CausalLMOutput modelOutput = (CausalLMOutput) this.predictor.predict(modelInput);
                NDArray allProbs = modelOutput.getLogits().get(":, -1, :", new Object[0]).softmax(1);
                modelInput = allProbs.topK(Math.toIntExact(numBeam), -1, true, false);
                NDArray outputIds = ((NDArray) modelInput.get(1)).expandDims(2);
                NDArray lastProbs = ((NDArray) modelInput.get(0)).normalize(1.0, 1L);

                assert outputIds.getShape().getShape().length == 3 : "Wrong shape";

                assert lastProbs.getShape().getShape().length == 2 : "Wrong Shape";

                attentionMask = attentionMask
                        .concat(manager.ones(new Shape(new long[] { numBatch, 1L }), DataType.INT64), -1).expandDims(1)
                        .repeat(1, numBeam);
                Function<NDArray, NDArray> fn = (ndarray) -> {
                    return ndarray.expandDims(1).repeat(1, numBeam);
                };
                NDList pastKeyValues = new NDList(
                        (Collection) modelOutput.getPastKeyValuesList().stream().map(fn).collect(Collectors.toList()));
                NDArray pastOutputIds = inputIds.expandDims(1).repeat(1, numBeam);
                searchState = new BeamBatchTensorList(outputIds, pastOutputIds, pastKeyValues, attentionMask,
                        lastProbs);
                numHeads = ((NDArray) pastKeyValues.get(0)).getShape().get(2);
                kvDim = ((NDArray) pastKeyValues.get(0)).getShape().getLastDimension();
            }

            NDScope ignore = new NDScope();

            try {
                long pastSeqLength = searchState.getPastOutputIds().getShape().getLastDimension();
                NDList modelInput = this.prepareInput(
                        searchState.getNextInputIds().reshape(new long[] { numBatch * numBeam, 1L }),
                        searchState.getPastAttentionMask().reshape(new long[] { numBatch * numBeam, -1L }),
                        pastSeqLength, this.config.getBeam());
                final long lnumHeads = numHeads;
                final long lkvDim = kvDim;
                Function<NDArray, NDArray> fn = (ndarray) -> {
                    return ndarray.reshape(new long[] { numBatch * numBeam, lnumHeads, pastSeqLength, lkvDim });
                };
                NDList pastKeyValues = new NDList(
                        (Collection) searchState.getPastKeyValues().stream().map(fn).collect(Collectors.toList()));
                modelInput.addAll(pastKeyValues);
                CausalLMOutput modelOutput = (CausalLMOutput) this.predictor.predict(modelInput);
                NDList generatedOutput = StepGeneration.beamStepGeneration(searchState.getLastProbs(),
                        modelOutput.getLogits(), numBatch, numBeam);
                searchState = updateSearchState(searchState, modelOutput, generatedOutput, manager);
                NDScope.unregister(new NDArray[] { searchState.getNextInputIds(), searchState.getPastOutputIds(),
                        searchState.getPastAttentionMask(), searchState.getLastProbs() });
                NDScope.unregister(searchState.getPastKeyValues());
            } catch (Throwable var26) {
                try {
                    ignore.close();
                } catch (Throwable var25) {
                    var26.addSuppressed(var25);
                }

                throw var26;
            }

            ignore.close();
            long[] outputIdsArray = searchState.getNextInputIds().toLongArray();

            for (int i = 0; i < this.endPosition.length; ++i) {
                long[] var31 = outputIdsArray;
                int var32 = outputIdsArray.length;

                for (int var33 = 0; var33 < var32; ++var33) {
                    long tokenId = var31[var33];
                    if (tokenId == this.config.getEosTokenId()) {
                        this.endPosition[i] = searchState.getPastOutputIds().getShape().get(1) + 1L;
                        break;
                    }
                }
            }
        } while (searchState.getPastOutputIds().getShape().getLastDimension()
                + 1L < (long) this.config.getMaxSeqLength());

        return searchState.getPastOutputIds().concat(searchState.getNextInputIds(), -1)
                .reshape(new long[] { numBatch * numBeam, -1L });
    }

    public NDArray contrastiveSearch(NDArray inputIds) throws TranslateException {
        this.endPosition = new long[Math.toIntExact(inputIds.getShape().get(0))];
        Arrays.fill(this.endPosition, (long) this.config.getMaxSeqLength());
        NDManager manager = inputIds.getManager();
        NDArray attentionMask = this.prepareAttentionMaskOffset(inputIds, this.config);
        ContrastiveBatchTensorList searchState = new ContrastiveBatchTensorList();

        do {
            NDArray candidateInputIds;
            if (searchState.getPastKeyValues() == null) {
                NDList modelInput = this.prepareInput(inputIds, attentionMask, 0L, 1);
                CausalLMOutput output = (CausalLMOutput) this.predictor.predict(modelInput);
                candidateInputIds = output.getLogits().get(":, -1, :", new Object[0]);
                searchState = new ContrastiveBatchTensorList(inputIds, attentionMask, output.getHiddenState(),
                        candidateInputIds, output.getPastKeyValuesList(), new long[0]);
            }

            NDScope ignore = new NDScope();

            try {
                NDArray topKIds = (NDArray) searchState.getLogits().topK(this.config.getK(), -1, true, false).get(1);
                candidateInputIds = topKIds.flatten().reshape(new long[] { -1L, 1L });

                assert candidateInputIds.getDataType() == DataType.INT64 : "inputIds datatype should be int64";

                assert candidateInputIds.getShape().getShape().length == 2 : "shape not right";

                NDList kCopyPastKeyValues = new NDList(
                        (Collection) searchState.getPastKeyValues().stream().map((ndarray) -> {
                            return ndarray.repeat(0, (long) this.config.getK());
                        }).collect(Collectors.toList()));

                assert ((NDArray) kCopyPastKeyValues.get(0)).getDataType() == DataType.FLOAT32
                        : "inputIds datatype should be Float32";

                long numBatch = topKIds.getShape().get(0);
                NDArray kCopyPastAttentionMask = searchState.getPastAttentionMask().repeat(0,
                        (long) this.config.getK());
                kCopyPastAttentionMask = kCopyPastAttentionMask.concat(manager
                        .ones(new Shape(new long[] { numBatch * (long) this.config.getK(), 1L }), DataType.INT64), 1);

                assert ((NDArray) kCopyPastKeyValues.get(0)).getShape().get(2) + 1L == kCopyPastAttentionMask.getShape()
                        .getLastDimension() : "attentionMask_seq = past_seq + new_input_seq";

                NDList candidateModelInput = this.prepareInput(candidateInputIds, kCopyPastAttentionMask,
                        searchState.getPastOutputIds().getShape().getLastDimension(), this.config.getK());
                candidateModelInput.addAll(kCopyPastKeyValues);
                CausalLMOutput candidateOutput = (CausalLMOutput) this.predictor.predict(candidateModelInput);
                NDList generatedOutput = StepGeneration.constrastiveStepGeneration(topKIds, searchState.getLogits(),
                        searchState.getPastHiddenStates(), candidateOutput.getHiddenState(), this.positionOffset,
                        this.config.getAlpha());
                searchState = updateSearchState(searchState, candidateOutput, generatedOutput, manager);
                NDScope.unregister(new NDArray[] { searchState.getPastOutputIds(), searchState.getPastAttentionMask(),
                        searchState.getLogits(), searchState.getPastHiddenStates() });
                NDScope.unregister(searchState.getPastKeyValues());
            } catch (Throwable var16) {
                try {
                    ignore.close();
                } catch (Throwable var15) {
                    var16.addSuppressed(var15);
                }

                throw var16;
            }

            ignore.close();
            long[] outputIdsArray = searchState.getPastOutputIds().toLongArray();

            for (int i = 0; i < this.endPosition.length; ++i) {
                long[] var21 = outputIdsArray;
                int var22 = outputIdsArray.length;

                for (int var23 = 0; var23 < var22; ++var23) {
                    long tokenId = var21[var23];
                    if (tokenId == this.config.getEosTokenId()) {
                        this.endPosition[i] = searchState.getPastOutputIds().getShape().get(1);
                        break;
                    }
                }
            }
        } while (searchState.getPastOutputIds().getShape().get(1) < (long) this.config.getMaxSeqLength());

        return searchState.getPastOutputIds();
    }

    private static BeamBatchTensorList updateSearchState(BeamBatchTensorList searchState, CausalLMOutput modelOutput,
            NDList generatedOutput, NDManager manager) {
        NDList pastKeyValues = searchState.getPastKeyValues();
        long numHeads = ((NDArray) pastKeyValues.get(0)).getShape().get(2);
        long kvDim = ((NDArray) pastKeyValues.get(0)).getShape().getLastDimension();
        long numBatch = searchState.getPastOutputIds().getShape().get(0);
        long numBeam = searchState.getPastOutputIds().getShape().get(1);
        long pastSeqLength = searchState.getPastOutputIds().getShape().getLastDimension();
        NDArray nextInputIds = (NDArray) generatedOutput.get(0);

        assert nextInputIds.getShape().getShape().length == 3 : "Wrong Shape";

        NDArray newProbs = (NDArray) generatedOutput.get(1);
        NDArray sourceBeamSelected = (NDArray) generatedOutput.get(2);
        NDIndex sourceBeamIndex = new NDIndex("{}, {}, ...",
                new Object[] {
                        manager.arange(0.0F, (float) numBatch, 1.0F, DataType.INT64).expandDims(1).repeat(1, numBeam),
                        sourceBeamSelected });
        NDArray pastOutputIds = searchState.getPastOutputIds().concat(searchState.getNextInputIds(), -1)
                .get(sourceBeamIndex);
        Function<NDArray, NDArray> fn = (ndarray) -> {
            return ndarray.reshape(new long[] { numBatch, numBeam, numHeads, pastSeqLength + 1L, kvDim })
                    .get(sourceBeamIndex);
        };
        pastKeyValues = new NDList(
                (Collection) modelOutput.getPastKeyValuesList().stream().map(fn).collect(Collectors.toList()));
        NDArray pastAttentionMask = searchState.getPastAttentionMask()
                .concat(manager.ones(new Shape(new long[] { numBatch, numBeam, 1L }), DataType.INT64), -1)
                .get(sourceBeamIndex);
        return new BeamBatchTensorList(nextInputIds, pastOutputIds, pastKeyValues, pastAttentionMask, newProbs);
    }

    private static ContrastiveBatchTensorList updateSearchState(ContrastiveBatchTensorList searchState,
            CausalLMOutput candidateOutput, NDList generatedOutput, NDManager manager) {
        assert candidateOutput.getLogits().getShape().get(1) == 1L
                : "dimension check: here, outputLogits corresponds to inputSeq == 1";

        long numBatch = searchState.getLogits().getShape().get(0);
        long logitsDim = searchState.getLogits().getShape().get(1);
        long pastSeqLengthPriorUpdate = searchState.getPastOutputIds().getShape().get(1);
        long numHeads = ((NDArray) searchState.getPastKeyValues().get(0)).getShape().get(1);
        long kvDim = ((NDArray) searchState.getPastKeyValues().get(0)).getShape().get(3);
        long hiddenDim = searchState.getPastHiddenStates().getShape().get(2);
        long k = candidateOutput.getLogits().getShape().get(0) / numBatch;
        NDArray select = (NDArray) generatedOutput.get(1);
        NDIndex selectIndex = new NDIndex("{}, {}, ...",
                new Object[] { manager.arange(0.0F, (float) numBatch, 1.0F, DataType.INT64), select.flatten() });
        NDArray nextLogits = candidateOutput.getLogits().reshape(new long[] { numBatch, k, logitsDim })
                .get(selectIndex);
        Function<NDArray, NDArray> fn = (ndarray) -> {
            return ndarray.reshape(new long[] { numBatch, k, numHeads, pastSeqLengthPriorUpdate + 1L, kvDim })
                    .get(selectIndex);
        };
        NDList nextPastKeyValue = new NDList(
                (Collection) candidateOutput.getPastKeyValuesList().stream().map(fn).collect(Collectors.toList()));
        NDArray newHiddenState = candidateOutput.getHiddenState();

        assert newHiddenState.getManager() == manager : "possible leaky memory";

        NDArray nextPastHiddenStates = searchState.getPastHiddenStates()
                .concat(newHiddenState.reshape(new long[] { numBatch, k, 1L, hiddenDim }).get(selectIndex), 1);
        NDArray outputIds = (NDArray) generatedOutput.get(0);
        NDArray nextOutputIds = searchState.getPastOutputIds().concat(outputIds, 1);
        NDArray nextPastAttentionMask = searchState.getPastAttentionMask()
                .concat(manager.ones(new Shape(new long[] { numBatch, 1L }), DataType.INT64), 1);
        return new ContrastiveBatchTensorList(nextOutputIds, nextPastAttentionMask, nextPastHiddenStates, nextLogits,
                nextPastKeyValue, new long[0]);
    }

    private NDArray prepareAttentionMaskOffset(NDArray inputIds, SearchConfig config) {
        boolean suffixPadding = config.isSuffixPadding();
        NDManager manager = inputIds.getManager();
        int numBatch = Math.toIntExact(inputIds.getShape().get(0));
        int initSeqSize = Math.toIntExact(inputIds.getShape().get(1));
        NDArray attentionMask = manager
                .ones(new Shape(new long[] { 1L, inputIds.getShape().getLastDimension() }), DataType.INT64)
                .reshape(new long[] { 1L, -1L }).repeat(0, (long) numBatch);
        long[][] offset = new long[numBatch][1];

        for (int i = 0; i < numBatch; ++i) {
            long[] aSequence = inputIds.get("{},:", new Object[] { i }).toLongArray();

            int idx;
            for (idx = 0; idx < initSeqSize && (!suffixPadding || aSequence[idx] != config.getPadTokenId())
                    && (suffixPadding || aSequence[idx] == config.getPadTokenId()); ++idx) {
            }

            attentionMask.set(new NDIndex("{},{}:{}",
                    new Object[] { i, suffixPadding ? idx : 0, suffixPadding ? initSeqSize : idx }), 0);
            if (!suffixPadding) {
                offset[i][0] = (long) idx;
            }
        }

        this.positionOffset = manager.create(offset);
        return attentionMask;
    }

    private NDList prepareInput(NDArray inputIds, NDArray attentionMask, long pastSeqLength, int repeat) {
        NDArray positionIds = inputIds
                .getManager().arange((float) pastSeqLength,
                        (float) (pastSeqLength + inputIds.getShape().getLastDimension()), 1.0F, DataType.INT64)
                .expandDims(0).repeat(0, inputIds.getShape().get(0));
        NDArray positionIdsShifted = positionIds.subi(this.positionOffset.repeat(0, (long) repeat));
        positionIds = positionIdsShifted.maximum(positionIdsShifted.zerosLike());
        inputIds.setName("input_ids");
        positionIds.setName("position_ids");
        attentionMask.setName("attention_mask");
        return new NDList(new NDArray[] { inputIds, attentionMask, positionIds });
    }

    public NDArray generate(NDArray inputIds) throws TranslateException {
        switch (this.searchName) {
            case "greedy":
                return this.greedySearch(inputIds);
            case "beam":
                return this.beamSearch(inputIds);
            case "contrastive":
                return this.contrastiveSearch(inputIds);
            default:
                throw new IllegalArgumentException(
                        "searchName not correctly specified. Please choose among: {greedy, beam, contrastive}");
        }
    }

    public NDArray getPositionOffset() {
        return this.positionOffset;
    }

    public long[] getEndPosition() {
        return this.endPosition;
    }
}

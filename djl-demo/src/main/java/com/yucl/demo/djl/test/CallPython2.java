package com.yucl.demo.djl.test;

import java.io.IOException;
import java.util.List;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

public class CallPython2 {
    public static void main(String[] args)
            throws TranslateException, ModelNotFoundException, MalformedModelException, IOException {
        BertTokenizer tokenizer = new BertTokenizer();
        List<String> tokenQ = tokenizer.tokenize(question.toLowerCase());
        List<String> tokenA = tokenizer.tokenize(resourceDocument.toLowerCase());
    }

}

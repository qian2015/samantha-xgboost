package org.grouplens.samantha.xgboost;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.grouplens.samantha.modeler.common.LearningInstance;
import org.grouplens.samantha.modeler.common.PredictiveModel;
import org.grouplens.samantha.modeler.featurizer.FeatureExtractor;
import org.grouplens.samantha.modeler.featurizer.FeatureExtractorUtilities;
import org.grouplens.samantha.modeler.featurizer.Featurizer;
import org.grouplens.samantha.modeler.model.IndexSpace;
import org.grouplens.samantha.modeler.featurizer.StandardFeaturizer;
import org.grouplens.samantha.modeler.instance.StandardLearningInstance;
import org.grouplens.samantha.server.config.ConfigKey;
import org.grouplens.samantha.server.exception.BadRequestException;
import org.grouplens.samantha.server.io.IOUtilities;
import play.libs.Json;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;

public class XGBoostModel implements PredictiveModel, Featurizer {
    final private StandardFeaturizer featurizer;
    final private IndexSpace indexSpace;
    private Booster booster;

    public XGBoostModel(IndexSpace indexSpace, List<FeatureExtractor> featureExtractors,
                        List<String> features, String labelName, String weightName) {
        this.featurizer = new StandardFeaturizer(indexSpace,
                featureExtractors, features, null, labelName, weightName);
        this.indexSpace = indexSpace;
    }

    public double[] predict(LearningInstance ins) {
        if (booster == null) {
            double[] preds = new double[1];
            preds[0] = 0.0;
            return preds;
        } else {
            List<LabeledPoint> list = new ArrayList<>(1);
            list.add(((XGBoostInstance) ins).getLabeledPoint());
            try {
                DMatrix data = new DMatrix(list.iterator(), null);
                float[][] rawPreds = booster.predict(data);
                double[] preds = new double[rawPreds[0].length];
                for (int i=0; i<preds.length; i++) {
                    preds[i] = rawPreds[0][i];
                }
                return preds;
            } catch (XGBoostError e) {
                throw new BadRequestException(e);
            }
        }
    }

    public List<ObjectNode> classify(List<ObjectNode> entities) {
        List<LearningInstance> instances = new ArrayList<>();
        for (JsonNode entity : entities) {
            instances.add(featurize(entity, true));
        }
        double[][] preds = predict(instances);
        List<ObjectNode> rankings = new ArrayList<>();
        for (int i=0; i<instances.size(); i++) {
            int k = preds[i].length;
            for (int j = 0; j < k; j++) {
                if (indexSpace.getKeyMapSize(ConfigKey.LABEL_INDEX_NAME.get()) > k) {
                    ObjectNode rec = Json.newObject();
                    rec.put("dataId", i);
                    String fea = (String) indexSpace.getKeyForIndex(
                            ConfigKey.LABEL_INDEX_NAME.get(), k);
                    IOUtilities.parseEntityFromStringMap(rec, FeatureExtractorUtilities.decomposeKey(fea));
                    rec.put("classProb", preds[i][k]);
                    rankings.add(rec);
                }
            }
        }
        return rankings;
    }

    public LearningInstance featurize(JsonNode entity, boolean update) {
        return new XGBoostInstance((StandardLearningInstance) featurizer.featurize(entity, update));
    }

    public void setXGBooster(Booster booster) {
        this.booster = booster;
    }

    public void saveModel(String modelFile) {
        try {
            this.booster.saveModel(modelFile);
        } catch (XGBoostError e) {
            throw new BadRequestException(e);
        }
    }

    public void loadModel(String modelFile) {
        try {
            ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(modelFile));
            this.booster = (Booster) inputStream.readUnshared();
        } catch (IOException | ClassNotFoundException e) {
            throw new BadRequestException(e);
        }
    }

    public void publishModel() {}
}

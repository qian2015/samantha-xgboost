package org.grouplens.samantha.xgboost;

import org.grouplens.samantha.modeler.featurizer.FeatureExtractor;
import org.grouplens.samantha.modeler.model.IndexSpace;
import org.grouplens.samantha.modeler.model.SpaceMode;
import org.grouplens.samantha.modeler.model.SpaceProducer;
import org.grouplens.samantha.modeler.tree.TreeKey;
import org.grouplens.samantha.server.config.ConfigKey;

import javax.inject.Inject;
import java.util.List;

public class XGBoostModelProducer {
    @Inject
    private SpaceProducer spaceProducer;

    @Inject
    public XGBoostModelProducer() {}

    private IndexSpace getIndexSpace(String spaceName, SpaceMode spaceMode) {
        IndexSpace indexSpace = spaceProducer.getIndexSpace(spaceName, spaceMode);
        indexSpace.requestKeyMap(TreeKey.TREE.get());
        indexSpace.requestKeyMap(ConfigKey.LABEL_INDEX_NAME.get());
        return indexSpace;
    }

    public XGBoostModel createXGBoostModel(String modelName, SpaceMode spaceMode,
                                           List<FeatureExtractor> featureExtractors,
                                           List<String> features, String labelName,
                                           String weightName) {
        IndexSpace indexSpace = getIndexSpace(modelName, spaceMode);
        return new XGBoostModel(indexSpace, featureExtractors, features, labelName, weightName);
    }
}

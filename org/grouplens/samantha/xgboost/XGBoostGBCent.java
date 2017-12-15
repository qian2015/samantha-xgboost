package org.grouplens.samantha.xgboost;

import org.grouplens.samantha.modeler.boosting.AbstractGBCent;
import org.grouplens.samantha.modeler.boosting.GBCentLearningInstance;
import org.grouplens.samantha.modeler.common.LearningInstance;
import org.grouplens.samantha.modeler.common.PredictiveModel;
import org.grouplens.samantha.modeler.featurizer.Feature;
import org.grouplens.samantha.modeler.featurizer.FeatureExtractor;
import org.grouplens.samantha.modeler.space.IndexSpace;
import org.grouplens.samantha.modeler.svdfeature.SVDFeature;

import java.util.List;

public class XGBoostGBCent extends AbstractGBCent implements PredictiveModel {

    public XGBoostGBCent(List<FeatureExtractor> treeExtractors,
                         List<String> treeFeatures, String labelName, String weightName,
                         IndexSpace indexSpace, SVDFeature svdFeature) {
        super(indexSpace, treeExtractors, treeFeatures, labelName, weightName, null, svdFeature);
    }

    public double[] predict(LearningInstance ins) {
        GBCentLearningInstance centIns = (GBCentLearningInstance) ins;
        double pred = svdfeaModel.predict(centIns.getSvdfeaIns())[0];
        for (Feature feature : centIns.getSvdfeaIns().getBiasFeatures()) {
            int idx = feature.getIndex();
            if (idx < trees.size()) {
                PredictiveModel tree = trees.get(idx);
                if (tree != null) {
                    pred += tree.predict(new XGBoostInstance(centIns.getTreeIns()))[0];
                }
            }
        }
        double[] preds = new double[1];
        preds[0] = pred;
        return preds;
    }

    public PredictiveModel getNumericalTree(int treeIdx) {
        return new XGBoostModel(indexSpace, featureExtractors, features, labelName, weightName);
    }

    public void publishModel() {}
}

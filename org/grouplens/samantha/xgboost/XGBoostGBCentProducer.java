package org.grouplens.samantha.xgboost;

import org.grouplens.samantha.modeler.featurizer.FeatureExtractor;
import org.grouplens.samantha.modeler.solver.ObjectiveFunction;
import org.grouplens.samantha.modeler.space.IndexSpace;
import org.grouplens.samantha.modeler.space.SpaceMode;
import org.grouplens.samantha.modeler.space.SpaceProducer;
import org.grouplens.samantha.modeler.svdfeature.SVDFeature;
import org.grouplens.samantha.modeler.svdfeature.SVDFeatureProducer;
import org.grouplens.samantha.modeler.tree.TreeKey;

import javax.inject.Inject;
import java.util.List;

public class XGBoostGBCentProducer {
    @Inject
    private SpaceProducer spaceProducer;

    @Inject
    private XGBoostGBCentProducer() {}

    public XGBoostGBCentProducer(SpaceProducer spaceProducer) {
        this.spaceProducer = spaceProducer;
    }

    public XGBoostGBCent createGBCent(String modelName, SpaceMode spaceMode,
                                      String labelName, String weightName,
                                      List<FeatureExtractor> svdfeaExtractors,
                                      List<FeatureExtractor> treeExtractors,
                                      List<String> biasFeas, List<String> ufactFeas,
                                      List<String> ifactFeas, List<String> treeFeas,
                                      int factDim, ObjectiveFunction objectiveFunction) {
        IndexSpace indexSpace = spaceProducer.getIndexSpace(modelName, spaceMode);
        indexSpace.requestKeyMap(TreeKey.TREE.get());
        SVDFeatureProducer svdfeaProducer = new SVDFeatureProducer(spaceProducer);
        SVDFeature svdfeaModel = svdfeaProducer.createSVDFeatureModel(modelName, spaceMode,
                biasFeas, ufactFeas, ifactFeas, labelName, weightName,
                null, svdfeaExtractors, factDim, objectiveFunction);
        return new XGBoostGBCent(treeExtractors, treeFeas, labelName, weightName,
                indexSpace, svdfeaModel);
    }

    public XGBoostGBCent createGBCentWithSVDFeatureModel(String modelName, SpaceMode spaceMode,
                                                         List<String> treeFeas,
                                                         List<FeatureExtractor> treeExtractors,
                                                         SVDFeature svdfeaModel) {
        IndexSpace indexSpace = spaceProducer.getIndexSpace(modelName, spaceMode);
        indexSpace.requestKeyMap(TreeKey.TREE.get());
        String labelName = svdfeaModel.getLabelName();
        String weightName = svdfeaModel.getWeightName();
        return new XGBoostGBCent(treeExtractors, treeFeas, labelName, weightName,
                indexSpace, svdfeaModel);
    }
}

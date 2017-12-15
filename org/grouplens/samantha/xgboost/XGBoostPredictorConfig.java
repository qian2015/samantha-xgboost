package org.grouplens.samantha.xgboost;

import com.fasterxml.jackson.databind.JsonNode;
import org.grouplens.samantha.modeler.common.LearningData;
import org.grouplens.samantha.modeler.featurizer.FeatureExtractor;
import org.grouplens.samantha.modeler.space.SpaceMode;
import org.grouplens.samantha.server.common.AbstractModelManager;
import org.grouplens.samantha.server.common.ModelManager;
import org.grouplens.samantha.server.config.ConfigKey;
import org.grouplens.samantha.server.expander.EntityExpander;
import org.grouplens.samantha.server.expander.ExpanderUtilities;
import org.grouplens.samantha.server.featurizer.FeatureExtractorConfig;
import org.grouplens.samantha.server.featurizer.FeatureExtractorListConfigParser;
import org.grouplens.samantha.server.featurizer.FeaturizerConfigParser;
import org.grouplens.samantha.server.io.RequestContext;
import org.grouplens.samantha.server.predictor.PredictiveModelBasedPredictor;
import org.grouplens.samantha.server.predictor.Predictor;
import org.grouplens.samantha.server.predictor.PredictorConfig;
import org.grouplens.samantha.server.predictor.PredictorUtilities;
import play.Configuration;
import play.inject.Injector;

import java.util.ArrayList;
import java.util.List;

public class XGBoostPredictorConfig implements PredictorConfig {
    private final String modelName;
    private final String modelFile;
    private final List<FeatureExtractorConfig> feaExtConfigs;
    private final List<String> features;
    private final String labelName;
    private final String weightName;
    private final Configuration daoConfigs;
    private final List<Configuration> expandersConfig;
    private final Injector injector;
    private final XGBoostMethod method;
    private final String daoConfigKey;
    private final String serializedKey;
    private final String insName;
    private final Configuration config;

    private XGBoostPredictorConfig(String modelName, List<FeatureExtractorConfig> feaExtConfigs,
                                   List<String> features, String labelName, String weightName,
                                   Configuration daoConfigs, List<Configuration> expandersConfig,
                                   Injector injector, XGBoostMethod method, String modelFile,
                                   String daoConfigKey, String insName, String serializedKey,
                                   Configuration config) {
        this.modelName = modelName;
        this.feaExtConfigs = feaExtConfigs;
        this.features = features;
        this.labelName = labelName;
        this.weightName = weightName;
        this.daoConfigs = daoConfigs;
        this.expandersConfig = expandersConfig;
        this.injector = injector;
        this.method = method;
        this.modelFile = modelFile;
        this.daoConfigKey = daoConfigKey;
        this.serializedKey = serializedKey;
        this.insName = insName;
        this.config = config;
    }

    public static PredictorConfig getPredictorConfig(Configuration predictorConfig,
                                                     Injector injector) {
        FeaturizerConfigParser parser = injector.instanceOf(
                FeatureExtractorListConfigParser.class);
        Configuration daoConfigs = predictorConfig.getConfig(ConfigKey.ENTITY_DAOS_CONFIG.get());
        List<FeatureExtractorConfig> feaExtConfigs = parser.parse(predictorConfig
                .getConfig(ConfigKey.PREDICTOR_FEATURIZER_CONFIG.get()));
        List<Configuration> expanders = ExpanderUtilities.getEntityExpandersConfig(predictorConfig);
        int round = predictorConfig.getInt("numTrees");
        return new XGBoostPredictorConfig(predictorConfig.getString("modelName"),
                feaExtConfigs, predictorConfig.getStringList("features"),
                predictorConfig.getString("labelName"),
                predictorConfig.getString("weightName"), daoConfigs, expanders, injector,
                new XGBoostMethod(predictorConfig.getConfig("methodConfig").asMap(), round),
                predictorConfig.getString("modelFile"),
                predictorConfig.getString("daoConfigKey"),
                predictorConfig.getString("instanceName"),
                predictorConfig.getString("serializedKey"), predictorConfig);
    }

    private class XGBoostModelManager extends AbstractModelManager {

        public XGBoostModelManager(String modelName, String modelFile, Injector injector) {
            super(injector, modelName, modelFile, null);
        }

        public Object createModel(RequestContext requestContext, SpaceMode spaceMode) {
            List<FeatureExtractor> featureExtractors = new ArrayList<>();
            for (FeatureExtractorConfig feaExtConfig : feaExtConfigs) {
                featureExtractors.add(feaExtConfig.getFeatureExtractor(requestContext));
            }
            XGBoostModelProducer producer = injector.instanceOf(XGBoostModelProducer.class);
            XGBoostModel model = producer.createXGBoostModel(modelName, spaceMode,
                    featureExtractors, features,
                    labelName, weightName);
            return model;
        }

        public Object buildModel(Object model, RequestContext requestContext) {
            JsonNode reqBody = requestContext.getRequestBody();
            XGBoostModel xgBoost = (XGBoostModel) model;
            LearningData learnData = PredictorUtilities.getLearningData(xgBoost, requestContext,
                    reqBody.get("learningDaoConfig"), daoConfigs,
                    expandersConfig, injector, true,
                    serializedKey, insName, labelName, weightName, null);
            LearningData validData = null;
            if (reqBody.has("validationDaoConfig")) {
                validData = PredictorUtilities.getLearningData(xgBoost, requestContext,
                        reqBody.get("validationDaoConfig"), daoConfigs,
                        expandersConfig, injector, true,
                        serializedKey, insName, labelName, weightName, null);
            }
            method.learn(xgBoost, learnData, validData);
            return model;
        }
    }

    public Predictor getPredictor(RequestContext requestContext) {
        ModelManager modelManager = new XGBoostModelManager(modelName, modelFile, injector);
        XGBoostGBCent model = (XGBoostGBCent) modelManager.manage(requestContext);
        List<EntityExpander> entityExpanders = ExpanderUtilities.getEntityExpanders(requestContext,
                expandersConfig, injector);
        return new PredictiveModelBasedPredictor(config, model, model,
                daoConfigs, injector, entityExpanders, daoConfigKey);
    }
}

package org.grouplens.samantha.xgboost;

import ml.dmlc.xgboost4j.LabeledPoint;
import org.grouplens.samantha.modeler.common.LearningData;
import org.grouplens.samantha.modeler.common.LearningInstance;
import play.Logger;

import java.util.Iterator;
import java.util.List;

public class XGBoostIterator implements Iterator {
    final private LearningData data;
    private int iter = 0;
    private XGBoostInstance instance;
    private List<LearningInstance> instances;
    private int num = 0;

    XGBoostIterator(LearningData data) {
        this.data = data;
    }

    @Override
    public LabeledPoint next() {
        XGBoostInstance back = instance;
        instance = null;
        num++;
        return back.getLabeledPoint();
    }

    @Override
    public boolean hasNext() {
        if (instance != null) {
            return true;
        }
        if (instances == null) {
            instances = data.getLearningInstance();
            iter = 0;
        } else if (instances.size() <= iter) {
            instances = data.getLearningInstance();
            iter = 0;
        }
        if (iter >= instances.size()) {
            Logger.info("Num of data points: {}", num);
            return false;
        } else {
            instance = (XGBoostInstance)instances.get(iter++);
            return true;
        }
    }
}

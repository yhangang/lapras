# coding:utf-8

import pandas as pd
import lapras
from datetime import datetime


def auto_model(df, target='target',to_drop=['id'], empty = 0.95, iv = 0.02, corr = 0.9, vif = False, method = 'mono',
               n_bins=8, min_samples=0.05, coef_negative = True, bins_show=False,iv_rank=False,  perform_show=True,
               pdo=40, rate=2, base_odds=1 / 60, base_score=600):
    ''' 自动化评分卡建模
    :param feature: 待分箱变量
    :param target: 目标特征
    :param n_bins: 最多分箱数
    :param min_sample: 每个分箱最小样本数
    :return: ScoreCard -- 评分卡实例
    '''

    start_time = datetime.now()
    # 变量筛选
    print("——开始初步筛选变量——")
    train_selected, dropped = lapras.select(df.drop(to_drop, axis=1), target=target, empty=empty, \
                                            iv=iv, corr=corr, vif=vif, return_drop=True, exclude=[])
    print("原始特征数：%s  筛选后特征数：%s" % (len(df.drop(to_drop, axis=1).columns)-1, len(train_selected.columns)-1))

    # 变量分箱
    print()
    print("——开始变量分箱——")
    c = lapras.Combiner()
    c.fit(train_selected, y=target, method=method, min_samples=min_samples, n_bins=n_bins)
    if bins_show:
        if iv_rank:
            cols = list(lapras.quality(train_selected, target=target).reset_index()['index'])
        else:
            cols = train_selected.columns
        for col in cols:
            if col != target:
                print(lapras.bin_stats(c.transform(train_selected[[col, target]], labels=True), col=col, target=target))
                lapras.bin_plot(c.transform(train_selected[[col, target]], labels=True), col=col, target=target)

    # 转换为WOE值
    print()
    print("——原始变量转换为WOE值——")
    transfer = lapras.WOETransformer()
    train_woe = transfer.fit_transform(c.transform(train_selected), train_selected[target], exclude=[target])

    # 再次变量筛选
    print()
    print("——再次筛选变量——")
    train_woe2, dropped = lapras.select(train_woe, target=target, empty=empty, \
                                            iv=iv, corr=corr, vif=vif, return_drop=True, exclude=[])
    print("原始特征数：%s  筛选后特征数：%s" % (len(train_woe.columns) - 1, len(train_woe2.columns) - 1))
    # 将woe转化后的数据做逐步回归
    # final_data = lapras.stepwise(train_woe, target=target, estimator='ols', direction='forward', criterion='aic',
    #                              exclude=[])
    final_data = train_woe2

    # 评分卡建模
    print()
    print("——评分卡建模——")

    model_cols = []
    if coef_negative == False:
        # 通过前向回归筛选入模变量，确保系数全部为正
        indexs = lapras.quality(final_data, target=target).index
        for index in list(indexs):
            model_cols.append(index)
            card = lapras.ScoreCard(
                combiner=c,
                transfer=transfer,
                pdo=pdo,
                rate=rate,
                base_odds=base_odds,
                base_score=base_score
            )
            card.fit(final_data[model_cols], final_data[target])
            if min(card.coef_) < 0:
                model_cols.remove(index)
    else:
        model_cols = list(final_data.drop([target], axis=1).columns)

    card = lapras.ScoreCard(
        combiner=c,
        transfer=transfer,
        pdo=pdo,
        rate=rate,
        base_odds=base_odds,
        base_score=base_score
    )

    card.fit(final_data[model_cols], final_data[target])
    print("intercept: %s" % card.intercept_)
    print("coef: %s" % card.coef_)

    # 输出模型效果
    print()
    print("——模型效果输出——")
    score = card.predict(final_data[model_cols])
    prob = card.predict_prob(final_data[model_cols])
    final_data['score'] = score
    final_data['prob'] = prob
    if perform_show:
        lapras.perform(prob, final_data[target])
        lapras.score_plot(final_data, score='score', target=target)

    print("KS: %.4f" % lapras.KS(prob, final_data[target]))
    print("AUC: %.4f" % lapras.AUC(prob, final_data[target]))
    print(lapras.LIFT(prob, final_data[target]))

    end_time = datetime.now()
    print()
    print("自动化建模完成,耗时：%s秒" % str((end_time - start_time).seconds))
    return card





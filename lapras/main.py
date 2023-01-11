# coding:utf-8

import pandas as pd
import lapras
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.max_colwidth = 100
pd.set_option('display.width',200)

'''
lapras项目主流程测试
'''

to_drop = ['id']
target = 'bad'
df = pd.read_csv('data/demo.csv',encoding="utf-8")
# print(lapras.detect(df.drop(to_drop,axis=1)))
iv_df = lapras.quality(df.drop(to_drop,axis=1),target = target)
print(iv_df)
print(iv_df.index)

train_selected, dropped = lapras.select(df.drop(to_drop,axis=1),target = target, empty = 0.99, \
                                                iv = 0.02, corr = 1,vif = 10, return_drop=True, exclude=[])
# train_selected = df.drop(to_drop,axis=1)
# print(dropped)
# print(train_selected.shape)
print(train_selected.columns)
# print(lapras.VIF(train_selected))

c = lapras.Combiner()
c.fit(train_selected, y = target, method = 'mono', min_samples = 0.05,n_bins=8)
# c.load(
#     {'feature1': [['nan'], ['良'], ['优']], 'feature2': [2.485, 19.105], 'feature3': [1.5, 2.5, 5.5, 9.5],
#      'feature4': [199.95, ], 'feature6': [51.145, 62.25, 70.065],
#      'feature7': [1.0,  82.65], 'feature9': [7.5, 104.0, 375.0, 5538.3],
#      'feature10': [2.5, 300.5,], 'feature11': [400.5],
#      'feature13': [3.5, 4.5, 10.5], 'feature14': [49.5, 218.0, 553.5]}
# )
print(c.export())

# print(c.transform(train_selected, labels=True).iloc[0:10, :])

# cols = list(lapras.quality(train_selected,target = target).reset_index()['index'])
# for col in cols:
#     if col != target:
#         print(lapras.bin_stats(c.transform(train_selected[[col, target]], labels=True), col=col, target=target))
#         lapras.bin_plot(c.transform(train_selected[[col,target]], labels=True), col=col, target=target)


# 转换为WOE值
transfer = lapras.WOETransformer()
train_woe = transfer.fit_transform(c.transform(train_selected), train_selected[target], exclude=[target])
# print(train_woe)
# print("PSI:%s" % lapras.PSI(df['age'], df['age']))
print("PPSI:", lapras.PPSI(df, df, feature='bad', target=target, return_frame=True))

# 将woe转化后的数据做逐步回归
final_data = lapras.stepwise(train_woe,target = target, estimator='ols', direction = 'both', criterion = 'aic', exclude = [])

# print(final_data.columns)
# final_data = train_woe

card = lapras.ScoreCard(
    combiner = c,
    transfer = transfer
)
col = list(final_data.drop([target],axis=1).columns)
# print(col)
card.fit(final_data[col], final_data[target])

score = card.predict(final_data[col])
prob = card.predict_prob(final_data[col])
final_data['score'] = score
final_data['prob'] = prob
# print("card.intercept_:" + str(card.intercept_))
# print("card.coef_:" + str(card.coef_))
# print(final_data[['score', 'prob']].iloc[:10,:])
#输出标准评分卡
print(card.export())
# print(lapras.F1(prob,final_data[target]))
# lapras.perform(prob,final_data[target])
# score_bond = [305, 460, 490, 520, 550, 580, 610, 640, 670, 700, 730, 760, 790, 820, 850, 880, 999]
# lapras.score_plot(final_data,score='score', target=target, output=True)
# print(lapras.LIFT(prob,final_data[target]))

# print(lapras.KS_bucket(final_data['score'], final_data[target], bucket=10, method = 'quantile'))


# if __name__ == "__main__":
#     df = pd.read_csv('data/model_data_demo.csv', encoding="utf")
#     card = lapras.auto_model(df,target='bad',to_drop=['employee_no'],bins_show=False,perform_show=True,empty = 0.95,
#                       iv = 0.02, corr = 0.9, vif = False, method = 'mono', n_bins=8, min_samples=0.05,
#                       coef_negative = False)
#     print(card.export())

# lapras.radar_plot([0.1,0.2,0.3,0.8,0.5],title='is am are',radar_labels=['111','222','333','444','请问'])

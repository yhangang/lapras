# coding:utf-8

import pandas as pd
import lapras

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.max_colwidth = 100
pd.set_option('display.width',200)

'''
lapras项目主流程测试
'''

# to_drop = ['employee_no']
to_drop = []
df = pd.read_csv('data/test_data.csv',encoding="utf-8")
print(lapras.detect(df))

print(lapras.quality(df,target = 'bad'))

train_selected, dropped = lapras.select(df.drop(to_drop,axis=1),target = 'bad', empty = 0.9, \
                                                iv = 0.02, corr = 0.9, return_drop=True, exclude=[])
# print(dropped)
# print(train_selected.shape)

c = lapras.Combiner()
c.fit(train_selected, y = 'bad', method = 'dt', min_samples = 0.05,n_bins=5)
# c.load({'A': [0.6584, 0.747, 0.8757, 0.9306], 'C': [0.1577, 0.2077, 0.2517, 0.4534], \
#         'D': [0.5133, 0.695, 0.8326, 0.9369], 'F': [8.0279, 41.0498, 48.468, 54.1201],\
#         'G': [['nan'], ['良', 'u', '个', '去', '好', '我', '是', '看', '额'], ['优']]})
# print(c.export())

# print(c.transform(train_selected, labels=True).iloc[0:10, :])

cols = train_selected.columns
# for col in cols:
#     if col != 'bad':
#         lapras.bin_plot(c.transform(train_selected[[col,'bad']], labels=True), col=col, target='bad')

# 转换为WOE值
transfer = lapras.WOETransformer()
train_woe = transfer.fit_transform(c.transform(train_selected), train_selected['bad'], exclude=['bad'])
# print(train_woe)
# print(lapras.metrics.PSI(df['C'], df['D']))

# 将woe转化后的数据做逐步回归
final_data = lapras.selection.stepwise(train_woe,target = 'bad', estimator='ols', direction = 'both', criterion = 'aic', exclude = [])

# print(final_data.columns)
# final_data = train_woe


card = lapras.ScoreCard(
    combiner = c,
    transfer = transfer,
)
col = list(final_data.drop(['bad'],axis=1).columns)
# print(col)
card.fit(final_data[col], final_data['bad'])

score = card.predict(final_data[col])
prob = card.predict_prob(final_data[col])
final_data['score'] = score
final_data['prob'] = prob
# print(card.intercept_)
# print(final_data[['score', 'prob']].iloc[:10,:])
#输出标准评分卡
print(card.export())
lapras.perform(prob,final_data['bad'])
# score_bond = [305, 460, 490, 520, 550, 580, 610, 640, 670, 700, 730, 760, 790, 820, 850, 880, 999]
lapras.score_plot(final_data,score='score', target='bad')
# print(lapras.LIFT(prob,final_data['bad']))

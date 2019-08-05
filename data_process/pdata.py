import pandas as pd
from data_process.tdata import tdata
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
class pdata(object):
    def __init__(self):
        return


    def pre_item(self, path):
        item_frame = pd.read_csv(path, sep='\t', header=None)
        item_frame.columns = ['item_id', 'cate_1_id', 'cate_id', 'brand_id', 'price']
        null_info = tdata.isnull(item_frame)
        return item_frame.loc[:, ['item_id', 'price']]

    def pre_user(self, path):
        user_frame = pd.read_csv(path, sep='\t')
        use_col = ['user_id', 'pred_gender', 'pred_age_level', 'pred_education_degre', 'pred_career_type', 'predict_income', 'pred_stage']
        user_frame.columns = use_col
        null_info = tdata.isnull(user_frame, True)
        # null_frame = {}
        # for key, val in null_info.items():
        #     if val > 0:
        #         null_frame[key] = user_frame.loc[user_frame[key].isnull()]
        #         col = 0
        #         fig, ax = plt.subplots(len(use_col[1:]))
        #         for c in list(set(use_col[1:]).difference(set(key))):
        #             freq = null_frame[key][c].value_counts()
        #             ax[col].barh(freq.index, freq.values)
        #             ax[col].set_yticks(freq.index)
        #             col += 1
        #         plt.waitforbuttonpress()
        #         #plt.close()
        fill_value = {}
        for col in use_col[3:]:
            if col in ['user_id', 'pred_gender', 'pred_age_level']:
                continue
            if col in ['pred_education_degre','pred_career_type', 'pred_stage']:
                fill_value[col] = user_frame.loc[:, col].mode()[0]
            else:
                fill_value[col] = user_frame.loc[:, col].mean()
        user_frame.fillna(fill_value, inplace=True)
        null_info = tdata.isnull(user_frame,True)

        pred_stage = user_frame['pred_stage'].unique()
        pred_career_type = user_frame['pred_career_type'].unique()
        pred_education_degre = user_frame['pred_education_degre'].unique()
        pred_age_level = user_frame['pred_age_level'].unique()

        pred_stage_freq = user_frame['pred_stage'].value_counts()
        pred_career_type_freq = user_frame['pred_career_type'].value_counts()
        pred_education_degre_freq = user_frame['pred_education_degre'].value_counts()
        pred_age_level_freq = user_frame['pred_age_level'].value_counts()

        
        return

    def pre_train(self, path):
        train_frame = pd.read_csv(path, sep='\t')
        train_frame.columns = ['user_id', 'item_id', 'behavior', 'date']
        #train_frame = tdata.down_sample(data_frame=train_frame, n=int(train_frame.shape[0]/100)).to_csv(path+'_sample', sep='\t',index=False)
        # group_cnt = {}
        # for c in list(train_frame.columns):
        #     group_cnt[c] = pd.value_counts(train_frame[c])
        #     print('{} - {}'.format(c, group_cnt[c].shape))
        user_freq = train_frame.loc[:,['user_id', 'item_id']].groupby(by=['user_id']).count().sort_values(by=['item_id'],ascending=False)
        user_freq.columns = ['user_count']
        user = [x for x in list(user_freq.index)]
        user_freq.insert(loc=user_freq.shape[1], column='user_id', value=user)
        user_freq.index = [x for x in range(user_freq.shape[0])]

        behavior_freq = train_frame.loc[:,['user_id', 'item_id', 'behavior']].groupby(by=['item_id', 'behavior']).count().sort_values(by=['user_id'],ascending=False)
        item_freq = train_frame.loc[:,['user_id', 'item_id']].groupby(by=['item_id']).count().sort_values(by=['user_id'],ascending=False)
        behavior_freq.columns = ['behavior_count']
        item_freq.columns = ['item_count']

        item = [x[0] for x in list(behavior_freq.index)]
        behavior = [x[1] for x in list(behavior_freq.index)]
        behavior_freq.insert(loc=behavior_freq.shape[1], column='item_id', value=item)
        behavior_freq.insert(loc=behavior_freq.shape[1], column='behavior', value=behavior)
        item = list(item_freq.index)
        item_freq.insert(loc=item_freq.shape[1], column='item_id', value=item)

        behavior_freq.index = [x for x in range(behavior_freq.shape[0])]
        item_freq.index = [x for x in range(item_freq.shape[0])]
        item_weight_frame = pd.merge(left=behavior_freq, right=item_freq, how='left', on='item_id')
        weight = item_weight_frame['behavior_count'] * item_weight_frame['item_count']
        item_weight_frame.insert(loc=item_weight_frame.shape[1], column='weight', value=weight)
        item_weight = item_weight_frame.loc[:,['weight', 'item_id']]
        item_weight_sum = item_weight.groupby(by=['item_id']).sum().sort_values(by=['item_weight'], ascending=False)
        item = list(item_weight_sum.index)
        item_weight_sum.insert(loc=item_weight_sum.shape[1], column='item_id', value=item)
        item_weight_sum.index = [x for x in range(item_weight_sum.shape[0])]
        train_frame_weight = pd.merge(left=train_frame, right=item_weight_sum, how='left', on='item_id')
        train_frame_weight = pd.merge(left=train_frame_weight, right=user_freq, how='left', on='user_id')
        # for name, group in train_frame.groupby(by=['item_id', 'behavior']):
        #     item = name[0]
        #     behavior = name[1]
        #     item_freq = item_behavior.get(item, {})
        #     if len(item_freq) == 0:
        #         item_freq = item_behavior[item] = {}
        #     item_freq[behavior] = item_freq.get(behavior, 0) + group.shape[0]
        #
        # print(item_behavior)
        return



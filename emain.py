from data_process.pdata import pdata
from mode.fm_mode import fm

if __name__ == '__main__':
    data = pdata()
    #data.pre_item('data/ECommAI_ubp_round1_item_feature')
    #data.pre_train('data/ECommAI_ubp_round1_train_sample')
    #data.pre_user('data/ECommAI_ubp_round1_user_feature')
    fm()

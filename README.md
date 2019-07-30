# EComm_AI

#数据集描述\n
#表格 1. 用户行为预测(test.tar.gz)\n
字段	类型	描述\n
user id	string	用户id\n
item_ids	string	预测会点击的商品集合(逗号分隔) 例如: 1,2,3,4,5

#表格 2. 用户历史行为(train.tar.gz)
字段	类型	描述
user id	int	用户id
item id	int	商品id
behavior type	string	行为
date	string	日期。格式： yyyymmdd

#表格 3. 用户特征数据(user_feature.tar.gz)
字段	类型	描述
user_id	int	用户 id
pred_gender	string	预测性别 eg : M(男性), F(女性)
pred_age_level	string	预测年龄段, eg: [35,39] 代表年龄位于35到39岁之间
pred_education_degree	int	预测教育程度
pred_career_type	int	预测职业
predict_income	float	预测收入
pred_stage	string	预测人生阶段。 每个人生阶段有一个独特的数字，比如婚姻中代表3，学生状态代表4，已育代表5，那么此字段为 3,4,5

#表格 4. 商品特征数据(item_feature.tar.gz)
字段	类型	描述
item_id	int	商品 id
cate_1_id	int	商品一级类目id
cate_id	int	商品叶子类目id
brand_id	int	商品品牌id
price	float	商品价格

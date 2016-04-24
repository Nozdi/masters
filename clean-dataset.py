import pandas as pd

from settings import (
    LR1_FEATURES,
    LR2_FEATURES,
    TIM_FEATURES,
    SM_FEATURES,
    ANN1_FEATURES,
    ANN2_1_FEATURES,
    ANN2_2_FEATURES,
    ANN3_FEATURES,
    OTHER_MODELS_RESUTLS,
    Y_NAME,
)

bad_data_ids = [
    'Sms100',
    'Sms212',
    'Sms098',
    'Sms160',
    'Sms131',
    'Sms101',
    'Sms210',
    'Sms213',
    'Sms061',
    'Sms118',
    'Sms152',
    'Sms156',
    'Sms136',
    'Sms146',
    'Sms220',
    'Sms192',
    'Sms107',
    'Sms126',
    'Sms171',
    'Sms151',
    'Sms148',
    'Sms219',
    'Sms204',
    'Sms095',
    'Sms137',
    'Sms166',
    'Sms105',
    'Sms208',
    'Sms201',
    'Sms141',
    'Sms097',
    'Sms183',
    'Sms089',
    'Sms119',
]

df = pd.read_excel('./data/sms-export.xlsx')
clean_df = df[~df['Name'].isin(bad_data_ids)].reset_index()

borderline = clean_df[clean_df['MalignancyCharacter'] == 2]
malignant = clean_df[clean_df['MalignancyCharacter'] == 1]
benign = clean_df[clean_df['MalignancyCharacter'] == 0]

borderline.describe().to_csv("./data/stats-borderline.csv")
malignant.describe().to_csv("./data/stats-malignant.csv")
benign.describe().to_csv("./data/stats-benign.csv")

stats = clean_df.describe().to_csv("./data/stats-all.csv")
clean_df.to_csv("./data/cleaned.csv", index=False)

# new features & preprocess
clean_df.loc[:, 'Menopause'] = pd.notnull(clean_df['MenopauseAge'])
all_dims = ['ADimension', 'BDimension', 'CDimension']
clean_df.loc[:, 'MaxDimension'] = clean_df[all_dims].max(axis=1)

idx = (pd.isnull(clean_df['PapBloodFlow']) & clean_df['Pap'] == 0)
clean_df.loc[idx, 'PapBloodFlow'] = 0
idx = (pd.isnull(clean_df['APapDimension']) & clean_df['Pap'] == 0)
clean_df.loc[idx, 'APapDimension'] = 0
idx = (pd.isnull(clean_df['SeptumThickness']) & clean_df['Septum'] == 0)
clean_df.loc[idx, 'SeptumThickness'] = 0


Papillarities = (
    (clean_df[['APapDimension', 'BPapDimension']].max(axis=1) > 3) | clean_df['Pap']
)
clean_df.loc[:, 'ANN2_Papillarities'] = Papillarities
clean_df.loc[:, 'ANN2_Bilateral'] = 0


Unilocular = (
    (clean_df['Echo'] == 1) | (clean_df['Echo'] == 2) |
    (clean_df['LoculesCount'] == 0) | (clean_df['LoculesCount'] == 1) |
    (clean_df['Solid'] == 1)
)

clean_df.loc[:, 'ANN2_Unilocular'] = Unilocular

clean_df.loc[:, 'ANN1_PSV'] = Unilocular


X_features = list(
    set(LR1_FEATURES + LR2_FEATURES + TIM_FEATURES + SM_FEATURES +
        ANN1_FEATURES + ANN2_1_FEATURES + ANN2_2_FEATURES + ANN3_FEATURES)
)

non_empty = clean_df[X_features + [Y_NAME] + OTHER_MODELS_RESUTLS].dropna()

non_empty.to_csv('./data/dataset.csv', index=False)

import pandas as pd

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


Unilocular = (
    (clean_df['Echo'] == 1) | (clean_df['Echo'] == 2) |
    (clean_df['LoculesCount'] == 0) | (clean_df['LoculesCount'] == 1) |
    (clean_df['Solid'] == 1)
)

clean_df.loc[:, 'ANN2_Unilocular'] = Unilocular


LR1_features = [
    'Age',
    'OvarianCancerInFamily',
    'HormoneReplacementTherapy',
    'PainAtExamination',
    'MaxDimension',
    'Ascites',
    'PapBloodFlow',
    'Solid',
    'ASolidDimension',
    'InnerWall',
    'Shadow',
    'Color',
]

LR2_features = [
    "Age",
    "Ascites",
    "PapBloodFlow",
    "ASolidDimension",
    "InnerWall",
    "Shadow",
]

Tim_features = [
    "Color",
    "Ca125",
    "APapDimension",
    "AgeAfterMenopause"
]

SM_features = [
    "Septum",
    "SmEchogenicity",
    "Location",
    "Ascites",
    "SmInnerWallThickness",
    "TumorVolume",
    "Solid",
    "Pap",
    "APapDimension",
    "InnerWall",
    "SeptumThickness",
    "AgeAfterMenopause"
]

ANN1_features = [
    'Age',
    'Menopause',
    'TumorVolume',
    'Pap',
    # '...'  # CALC & ADD PSV
]


ANN2_1_features = [
    'ANN2_Papillarities',
    'Color',
    'Menopause',
    'Ca125'
]

ANN2_2_features = [
    'ANN2_Papillarities',
    'InnerWall',
    'ANN2_Unilocular',
    'Ascites',
    'Menopause',
    'Ca125',
]

ANN3 = [
    # ''
]


X_features = list(
    set(LR1_features + LR2_features + Tim_features + SM_features +
        ANN1_features + ANN2_1_features + ANN2_2_features)
)
y_name = 'MalignancyCharacter'


clean_df.loc[:, 'GiradsDiagBin'] = clean_df['GiradsDiag'] > 3

other_models_results = [
    'TimmermannBin',
    'LR1Bin',
    'LR2Bin',
    'SMBin',
    'GiradsDiagBin',
    'AdnexBin',
]

non_empty = clean_df[X_features + [y_name] + other_models_results].dropna()

non_empty.to_csv('./data/dataset.csv', index=False)

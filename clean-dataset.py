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
clean_df = df[~df['Name'].isin(bad_data_ids)]

borderline = clean_df[clean_df['MalignancyCharacter'] == 2]
malignant = clean_df[clean_df['MalignancyCharacter'] == 1]
benign = clean_df[clean_df['MalignancyCharacter'] == 0]

borderline.describe().to_csv("./data/stats-borderline.csv")
malignant.describe().to_csv("./data/stats-malignant.csv")
benign.describe().to_csv("./data/stats-benign.csv")

stats = clean_df.describe().to_csv("./data/stats-all.csv")
clean_df.to_csv("./data/cleaned.csv", index=False)

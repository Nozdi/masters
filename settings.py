
LR1_FEATURES = [
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

LR2_FEATURES = [
    "Age",
    "Ascites",
    "PapBloodFlow",
    "ASolidDimension",
    "InnerWall",
    "Shadow",
]

TIM_FEATURES = [
    "Color",
    "Ca125",
    "APapDimension",
    "AgeAfterMenopause"
]

SM_FEATURES = [
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

ANN1_FEATURES = [
    'Age',
    'Menopause',
    'TumorVolume',
    'Pap',
    # 'ANN1_PSV',
]


ANN2_1_FEATURES = [
    'Menopause',
    'Ca125',
    'Color',
    'ANN2_Papillarities',
]

ANN2_2_FEATURES = [
    'Menopause',
    'Ca125',
    'Ascites',
    'ANN2_Unilocular',
    'ANN2_Smooth',
    'ANN2_Papillarities',
    'ANN2_Bilateral',
]

ANN3_FEATURES = [
    'log_Ca125',
    'Ultrasound',
    'Age',
]


OTHER_MODELS_RESUTLS = [
    'TimmermannBin',
    'LR1Bin',
    'LR2Bin',
    'SMBin',
    'AdnexBin',
    'ann_2_1_Bin',
    'ann_2_2_Bin',
]

Y_NAME = 'MalignancyCharacter'

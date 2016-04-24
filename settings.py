
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
    'ANN1_PSV',
]


ANN2_1_FEATURES = [
    'ANN2_Papillarities',
    'Color',
    'Menopause',
    'Ca125'
]

ANN2_2_FEATURES = [
    'ANN2_Papillarities',
    'InnerWall',
    'ANN2_Unilocular',
    'Ascites',
    'ANN2_Bilateral',
    'Menopause',
    'Ca125',
]

ANN3_FEATURES = [
    # ''
]


OTHER_MODELS_RESUTLS = [
    'TimmermannBin',
    'LR1Bin',
    'LR2Bin',
    'SMBin',
    'AdnexBin',
]

Y_NAME = 'MalignancyCharacter'

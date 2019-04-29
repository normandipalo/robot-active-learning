hyperp = {"INITIAL_TRAIN_EPS" : 150,

"BC_LR" : 1e-3,
"BC_HD" : 64,
"BC_HL" : 2,
"BC_BS" : 128,
"BC_EPS" : 150,

"AE_HD" : 32,
"AE_HL" : 2,
"AE_LR" : 1e-3,
"AE_BS" : 64,
"AE_EPS" : 20,

"TEST_EPS" : 10,
"ACTIVE_STEPS_RETRAIN" : 1,
"ACTIVE_ERROR_THR" : 1.1,

"ORG_TRAIN_SPLIT" : .5}

INITIAL_TRAIN_EPS = hyperp["INITIAL_TRAIN_EPS"]
BC_LR = hyperp["BC_LR"]
BC_HD = hyperp["BC_HD"]
BC_HL = hyperp["BC_HL"]
BC_BS = hyperp["BC_BS"]
BC_EPS = hyperp["BC_EPS"]

AE_HD = hyperp["AE_HD"]
AE_HL = hyperp["AE_HL"]
AE_LR = hyperp["AE_LR"]
AE_BS = hyperp["AE_BS"]
AE_EPS = hyperp["AE_EPS"]

TEST_EPS = hyperp["TEST_EPS"]
ACTIVE_STEPS_RETRAIN = hyperp["ACTIVE_STEPS_RETRAIN"]
ACTIVE_ERROR_THR = hyperp["ACTIVE_ERROR_THR"]

ORG_TRAIN_SPLIT = hyperp["ORG_TRAIN_SPLIT"]

hyperp = {"INITIAL_TRAIN_EPS" : 30,

"BC_LR" : 1e-3,
"BC_HD" : 128,
"BC_HL" : 2,
"BC_BS" : 64,
"BC_EPS" : 400,

"AE_HD" : 8,
"AE_HL" : 2,
"AE_LR" : 1e-3,
"AE_BS" : 64,
"AE_EPS" : 20,

"TEST_EPS" : 100,
"ACTIVE_STEPS_RETRAIN" : 3,
"ACTIVE_ERROR_THR" : 1.15,

"ORG_TRAIN_SPLIT" : .5,
"FULL_TRAJ_ERROR" : False,
"CTRL_NORM" : True,
"RENDER_TEST" : False}

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
FULL_TRAJ = hyperp["FULL_TRAJ_ERROR"]
CTRL_NORM = hyperp["CTRL_NORM"]

TEST_EPS = hyperp["TEST_EPS"]
ACTIVE_STEPS_RETRAIN = hyperp["ACTIVE_STEPS_RETRAIN"]
ACTIVE_ERROR_THR = hyperp["ACTIVE_ERROR_THR"]

ORG_TRAIN_SPLIT = hyperp["ORG_TRAIN_SPLIT"]
RENDER_TEST = hyperp["RENDER_TEST"]

hyperp = {"INITIAL_TRAIN_EPS" : 40,

"BC_LR" : 1e-3,
"BC_HD" : 128,
"BC_HL" : 2,
"BC_BS" : 64,
"BC_EPS" : 500,

"AE_HD" : 8,
"AE_HL" : 2,
"AE_LR" : 1e-3,
"AE_BS" : 64,
"AE_EPS" : 20,

"TEST_EPS" : 50,
"ACTIVE_STEPS_RETRAIN" : 5,
"ACTIVE_ERROR_THR" : 3.,

"ORG_TRAIN_SPLIT" : 0.75,
"FULL_TRAJ_ERROR" : True,
"CTRL_NORM" : True,
"RENDER_TEST" : False,
"RENDER_ACT_EXP" : False,
"TAKE_MAX" : True,
"AE_RESTART" : True,
"MAX_ACT_STEPS" : 100,
"DISABLE_NORM" : False}

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
AE_RESTART = hyperp["AE_RESTART"]

TEST_EPS = hyperp["TEST_EPS"]
ACTIVE_STEPS_RETRAIN = hyperp["ACTIVE_STEPS_RETRAIN"]
ACTIVE_ERROR_THR = hyperp["ACTIVE_ERROR_THR"]

ORG_TRAIN_SPLIT = hyperp["ORG_TRAIN_SPLIT"]
RENDER_TEST = hyperp["RENDER_TEST"]
RENDER_ACT_EXP = hyperp["RENDER_ACT_EXP"]
TAKE_MAX = hyperp["TAKE_MAX"]
MAX_ACT_STEPS = hyperp["MAX_ACT_STEPS"]
DISABLE_NORM = hyperp["DISABLE_NORM"]

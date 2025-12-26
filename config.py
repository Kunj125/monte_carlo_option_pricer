S0 = 100.0
K = 100.0
R = 0.05
# SIGMA = 0.2
T = 1.0
STEPS = 50

# heston params
V0 = 0.04
KAPPA = 2.0
THETA = 0.04
XI = 0.3  # volat of volat
RHO = -0.7  # Correlation (stock down -> volat up)

BATCH_SIZE = 256
EPOCHS = 2000
RISK_AVERSION = 1.0
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "heston_model.pth"

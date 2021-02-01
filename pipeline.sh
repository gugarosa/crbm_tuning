# Common variables definition
DATA="natural_images"
BATCH_SIZE=4
EPOCHS=1
DEVICE="cpu"
N_RUNS=1

# Architecture variables
VISIBLE_SHAPE=28
N_CHANNELS=1
N_CLASSES=10
STEPS=1

# Optimization variables
MH="ga"
N_AGENTS=1
N_TERMINALS=2
N_ITER=1
MIN_DEPTH=1
MAX_DEPTH=5

# Iterates through all possible seeds
for SEED in $(seq 1 $N_RUNS); do
    # Optimizes an architecture
    python crbm_optimization.py ${DATA} ${MH} -visible_shape ${VISIBLE_SHAPE} ${VISIBLE_SHAPE} -n_channels ${N_CHANNELS} -steps ${STEPS} -batch_size ${BATCH_SIZE} -epochs ${EPOCHS} -device ${DEVICE} -n_agents ${N_AGENTS} -n_iter ${N_ITER} -seed ${SEED}
    # python crbm_tree_optimization.py ${DATA} ${MH} -visible_shape ${VISIBLE_SHAPE} ${VISIBLE_SHAPE} -n_channels ${N_CHANNELS} -steps ${STEPS} -batch_size ${BATCH_SIZE} -epochs ${EPOCHS} -device ${DEVICE} -n_trees ${N_AGENTS} -n_terminals ${N_TERMINALS} -n_iter ${N_ITER} -min_depth ${MIN_DEPTH} -max_depth ${MAX_DEPTH} -seed ${SEED}

    # Evaluates an optimized architecture
    python crbm_evaluation.py ${DATA} ${MH}.pkl -visible_shape ${VISIBLE_SHAPE} ${VISIBLE_SHAPE} -n_channels ${N_CHANNELS} -n_classes ${N_CLASSES} -steps ${STEPS} -batch_size ${BATCH_SIZE} -epochs ${EPOCHS} -device ${DEVICE} -seed ${SEED}

    # Stores files in the outputs folder
    mv ${MH}.pkl outputs/crbm_${MH}_${SEED}.pkl
    mv crbm.pth outputs/crbm_${MH}_${SEED}.pth
    mv crbm_fine_tuned.pth outputs/crbm_fine_tuned_${MH}_${SEED}.pth
done
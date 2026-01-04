#!/bin/bash

##############################################################################
# Run DNC or LSTM sweeps over HPPs
##############################################################################

# 1) Kill existing screen sessions
echo "[INFO] Killing existing screen sessions named '*'..."
screen -ls | grep '\.' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

echo "[INFO] Starting runs..."


########## RUN PREFIX #############
RUN_PREFIX="copy_" 


# Define hyperparameter arrays (adjust these values as needed):
TASKS=("copy") # can use copy, reverse, sort, add, penn_treebank, etc..
# ARCHITECTURES=("Transformer" "Mamba" "SSM") # can use DNC and LSTM only right now
ARCHITECTURES=("DNC")

MODEL_SCALES=(8 16 32 64) # Can choose anything here from 1 to 64: MODEL_SCALES=(1 2 4 8 16 32 64) 
# MODEL_SCALES=(64)
hidden_size=128
memory_size=128
head_size=128
num_heads=1
input_dim=128
variance_reduction=1.

INPUT_SAMPLE_LENGTHS=(100)
MICRO_BATCH_SIZES=(1)
MACRO_BATCH_SIZES=(1)

# LEARNING_RATES=(0.05 0.01 0.005 0.001) 
# LEARNING_RATES=(0.05 0.01 0.005)  # sweep LRs from 1e-5 to .1 here 
EPSILONS=(0.01 0.001 0.0001)  # sweep LRs from 1e-5 to .1 here 
# LEARNING_RATES=(0.1)  # sweep LRs from 1e-5 to .1 here 
EPSILONS=(0.1)  # sweep LRs from 1e-5 to .1 here 

MAX_NUMS=(120)
# WARMUP_STEPS=(0) TO IMPLEMENT
WEIGHT_DECAYS=(0)
GRAD_CLIPS=(0)
SOLVERS=("1.5-SPSA") # "BPTT" "1SPSA" "1.5-SPSA" "2SPSA" "1.5-SPSA","Sanger-SPSA"

BETA1s=(0.)
BETA2s=(0.)
PROBE_PROCONDITIONINGS=(false)

# SANGER_RANKS=(1 2 4 8 16)
SANGER_RANKS=(1)
alpha_eye_scalars=(1.0)
beta_eigen_sangers=(0)

NUM_PERTURBATIONS=(8)
# NUM_PERTURBATIONS=(8 96 512)
saturating_alphas=(0.5 0.1)

OVERFITS=(true)

# Other configurations:
LOG_INTERVAL=100
MAX_ITERS=5000

TIE_EPS_TO_LR=true # true if you want LR to override EPS so they are equal
ADAM=false # true if you want to use adam and BPTT, otherwise set to false. Not implemented for ZOO.
WANDB=false # true if you want to log results to wandb

WANDB_PROJ=""

# Function to truncate a long session name (if needed)
truncate_name() {
  echo "$1" | cut -c1-65
}

# --- Detect Available GPUs ---
echo "[INFO] Detecting available GPUs..."
GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
NUM_GPUS=${#GPU_IDS[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No GPUs detected by nvidia-smi. Exiting."
    exit 1
fi
echo "[INFO] Detected ${NUM_GPUS} GPUs with IDs: ${GPU_IDS[@]}"

run_counter=0

# --- Loop over the original hyperparameters ---
for TASK in "${TASKS[@]}"; do
    for ARCH in "${ARCHITECTURES[@]}"; do
        for SOLVER in "${SOLVERS[@]}"; do
            for MODEL_SCALE in "${MODEL_SCALES[@]}"; do
                for INPUT_SAMPLE_LENGTH in "${INPUT_SAMPLE_LENGTHS[@]}"; do
                    for MICRO_BS in "${MICRO_BATCH_SIZES[@]}"; do
                        for MACRO_BS in "${MACRO_BATCH_SIZES[@]}"; do
                            for LR in "${LEARNING_RATES[@]}"; do
                                for EPS in "${EPSILONS[@]}"; do
                                for PROBE_PROCONDITIONING in "${PROBE_PROCONDITIONINGS[@]}"; do
                                for BETA1 in "${BETA1s[@]}"; do
                                for BETA2 in "${BETA2s[@]}"; do
                                for SANGER_RANK in "${SANGER_RANKS[@]}"; do
                                for beta_eigen_sanger in  "${beta_eigen_sangers[@]}"; do
                                for saturating_alpha in "${saturating_alphas[@]}"; do
                                for alpha_eye_scalar in "${alpha_eye_scalars[@]}"; do
                                    for MAX_NUM in "${MAX_NUMS[@]}"; do
                                        for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                                            for GRAD_CLIP in "${GRAD_CLIPS[@]}"; do
                                                for OVERFIT in "${OVERFITS[@]}"; do
                                                
                                                  # --- Compute model scale–dependent parameters ---
                                                  # this_hidden_size=$(( hidden_size * MODEL_SCALE ))
                                                  # this_memory_size=$(( memory_size * MODEL_SCALE ))
                                                  # this_head_size=$(( head_size * MODEL_SCALE ))
                                                  # # this_num_head=$(( num_heads * MODEL_SCALE ))
                                                  # this_num_head=$(( num_heads * 1 ))
                                                  # this_input_size=$(( input_dim * MODEL_SCALE ))
                                                  this_hidden_size=$(( hidden_size * MODEL_SCALE ))
                                                  this_memory_size=$(( memory_size * MODEL_SCALE ))
                                                  this_head_size=$(( head_size * MODEL_SCALE ))
                                                  # this_num_head=$(( num_heads * MODEL_SCALE ))
                                                  this_num_head=$(( num_heads * 1 ))
                                                  this_input_size=$(( input_dim * MODEL_SCALE ))
                                                  
                                                  # --- Loop over new ablation parameters ---
                                                  for numPert in "${NUM_PERTURBATIONS[@]}"; do
                                                      

                                                    EXTRA_FLAGS=""
                                                    if [ "$PROBE_PROCONDITIONING" = true ]; then
                                                      EXTRA_FLAGS+=" --use_probe_preconditioning"
                                                    fi
                                                    
                                                    if [ "$ADAM" = true ]; then
                                                      EXTRA_FLAGS+=" --use_adam"
                                                    fi

                                                    if [ "$TIE_EPS_TO_LR" = true ]; then
                                                       EPS=$LR
                                                    fi
                                                    
                                                    if [ "$OVERFIT" = true ]; then
                                                      EXTRA_FLAGS+=" --overfit_to_one_batch_flag"
                                                    fi
                                                    
                                                    # if [ "$ADAPTIVE" = true ]; then
                                                    #   EXTRA_FLAGS+=" --adaptive"
                                                    # fi
    
                                                    if [ "$WANDB" = true ]; then
                                                      EXTRA_FLAGS+=" --wandb"
                                                      EXTRA_FLAGS+=" --wandb_proj ${WANDB_PROJ}"
                                                      EXTRA_FLAGS+=" --wandb_run_name ${RUN_NAME_BASE}"
                                                    fi
    
                                                        
                                                    RUN_NAME_BASE="${RUN_PREFIX}_${ARCH}_pert_${numPert}_scale${MODEL_SCALE}_SOLV_${SOLVER}_lr${LR}_overfit_${OVERFIT}_cx${INPUT_SAMPLE_LENGTH}_maxnum${MAX_NUM}_hs${this_hidden_size}_mem${this_memory_size}_head${this_head_size}"
        
                                                    # --- Assign GPU for this run ---
                                                    gpu_index=$(( run_counter % NUM_GPUS ))
                                                    assigned_gpu_id=${GPU_IDS[$gpu_index]}
                                                    device_string="cuda:${assigned_gpu_id}" # Device string for python script
                                                    # --- END GPU Assignment ---
                                                    
                                                    RUN_NAME=$(truncate_name "${RUN_NAME_BASE}")
                                                    echo "[INFO] Launching screen session: $RUN_NAME_BASE"
                                                    
                                                    screen -dmS "$RUN_NAME" bash -c "
                                                    echo '[INFO] Starting run: $RUN_NAME';
                                                    export WANDB_RUN_NAME=$RUN_NAME;
                                                    python rge_series_experiments.py \
                                                          --model_type ${ARCH} \
                                                          --device ${device_string} \
                                                          --task ${TASK} \
                                                          --seq_length ${INPUT_SAMPLE_LENGTH} \
                                                          --hidden_size ${this_hidden_size} \
                                                          --memory_size ${this_memory_size} \
                                                          --head_size ${this_head_size} \
                                                          --num_heads ${this_num_head} \
                                                          --input_size ${this_input_size} \
                                                          --micro_batch_size ${MICRO_BS} \
                                                          --macro_batch_size ${MACRO_BS} \
                                                          --max_iterations ${MAX_ITERS} \
                                                          --log_interval ${LOG_INTERVAL} \
                                                          --learning_rate ${LR} \
                                                          --epsilon ${EPS} \
                                                          --sanger_rank ${SANGER_RANK} \
                                                          --weight_decay ${WEIGHT_DECAY} \
                                                          --max_num ${MAX_NUM} \
                                                          --grad_clip ${GRAD_CLIP} \
                                                          --num_perturbations ${numPert} \
                                                          --tokenizer char_level \
                                                          --distribution rad \
                                                          --beta1 ${BETA1} \
                                                          --beta2 ${BETA2} \
                                                          --solver ${SOLVER} \
                                                          --sanger_qr_every 100 \
                                                          --saturating_alpha ${saturating_alpha} \
                                                          --warmup_iters 1 \
                                                          --seed 42 \
                                                          --alpha_eye_scalar ${alpha_eye_scalar} \
                                                          --beta_eigen_sanger ${beta_eigen_sanger} \
                                                          --output_dir ./results_15SPSA_run \
                                                          ${EXTRA_FLAGS} \
                                                          ;
                                                    echo '[INFO] Finished run: $RUN_NAME_BASE';
                                                    exec bash
                                                    "
                                                    run_counter=$(( run_counter + 1 ))
                                                    # sleep 1
                                                   
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                                done
                                done
                                done
                                done
                                done
                                done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "[INFO] Done launching all screen sessions."

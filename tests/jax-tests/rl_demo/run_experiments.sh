#!/bin/bash
# Experiment runner for REINFORCE demo parameter estimation
# Run different configurations and compare results

echo "REINFORCE Demo Experiment Runner"
echo "================================"
echo ""

# Configuration
NUM_UPDATES=500
BATCH_SIZE=128

# Experiment 1: Baseline (Phase 1-3 only)
echo "Experiment 1: Baseline (Phase 1-3 improvements only)"
echo "------------------------------------------------------"
python reinforce_demo.py \
    --num-updates $NUM_UPDATES \
    --batch-size $BATCH_SIZE \
    2>&1 | tee logs/exp1_baseline.log

echo ""
echo "Experiment 1 complete!"
echo ""

# Experiment 2: With supervised parameter loss
echo "Experiment 2: + Direct parameter loss"
echo "------------------------------------------------------"
python reinforce_demo.py \
    --num-updates $NUM_UPDATES \
    --batch-size $BATCH_SIZE \
    --param-loss-coeff 0.1 \
    2>&1 | tee logs/exp2_param_loss.log

echo ""
echo "Experiment 2 complete!"
echo ""

# Experiment 3: With all advanced features
echo "Experiment 3: All advanced features enabled"
echo "------------------------------------------------------"
python reinforce_demo.py \
    --num-updates $NUM_UPDATES \
    --batch-size $BATCH_SIZE \
    --param-loss-coeff 0.1 \
    --waveform-bonus-coeff 0.5 \
    --use-multiscale-loss \
    --entropy-decay \
    --concentration-schedule \
    2>&1 | tee logs/exp3_all_features.log

echo ""
echo "Experiment 3 complete!"
echo ""

# Experiment 4: Aggressive parameter matching
echo "Experiment 4: Aggressive parameter matching (high param loss)"
echo "------------------------------------------------------"
python reinforce_demo.py \
    --num-updates $NUM_UPDATES \
    --batch-size $BATCH_SIZE \
    --param-loss-coeff 0.3 \
    --waveform-bonus-coeff 1.0 \
    --use-multiscale-loss \
    --entropy-decay \
    --concentration-schedule \
    2>&1 | tee logs/exp4_aggressive.log

echo ""
echo "Experiment 4 complete!"
echo ""

# Experiment 5: Long training
echo "Experiment 5: Long training (1000 updates)"
echo "------------------------------------------------------"
python reinforce_demo.py \
    --num-updates 1000 \
    --batch-size $BATCH_SIZE \
    --param-loss-coeff 0.2 \
    --waveform-bonus-coeff 0.5 \
    --use-multiscale-loss \
    --entropy-decay \
    --concentration-schedule \
    2>&1 | tee logs/exp5_long.log

echo ""
echo "Experiment 5 complete!"
echo ""

echo "================================"
echo "All experiments complete!"
echo ""
echo "Compare results in logs/ directory:"
echo "  - exp1_baseline.log"
echo "  - exp2_param_loss.log"
echo "  - exp3_all_features.log"
echo "  - exp4_aggressive.log"
echo "  - exp5_long.log"
echo ""
echo "Look for final test metrics:"
echo "  - Continuous MAE (target: < 0.05)"
echo "  - Categorical Accuracy (target: > 80%)"
echo "  - Spectral Distance (target: < 1.5)"

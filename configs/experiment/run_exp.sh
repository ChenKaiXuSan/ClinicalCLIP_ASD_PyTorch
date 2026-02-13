#!/bin/bash
#
# Quick experiment runner for ablation studies
# Usage:
#   ./run_exp.sh all          # Run all experiments (B1-B4, C1-C3, D)
#   ./run_exp.sh B            # Run B-series only (B1-B4)
#   ./run_exp.sh C            # Run C-series only (C1-C3)
#   ./run_exp.sh D            # Run D-series only
#   ./run_exp.sh B1 B2 B3     # Run specific experiments
#

set -e

# Always run from project root
cd /workspace/code/ClinicalCLIP_ASD_PyTorch
conda activate /opt/conda/envs/clip

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# All experiments
ALL_B=("B1_clip_only" "B2_map_only" "B3_full" "B4_full_token")
ALL_C=("C1_channel_gate" "C2_weighted_pool" "C3_sigmoid_gate")
ALL_D=("D_retrieval")

EXPERIMENTS=()

# Parse arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: ./run_exp.sh [all|B|C|D|experiment_names...]"
    echo ""
    echo "Examples:"
    echo "  ./run_exp.sh all                    # Run all experiments"
    echo "  ./run_exp.sh B                      # Run B1-B4"
    echo "  ./run_exp.sh C                      # Run C1-C3"
    echo "  ./run_exp.sh D                      # Run D"
    echo "  ./run_exp.sh B1 B2                  # Run specific experiments"
    exit 1
fi

for arg in "$@"; do
    case "$arg" in
        all)
            EXPERIMENTS+=("${ALL_B[@]}" "${ALL_C[@]}" "${ALL_D[@]}")
            ;;
        B)
            EXPERIMENTS+=("${ALL_B[@]}")
            ;;
        C)
            EXPERIMENTS+=("${ALL_C[@]}")
            ;;
        D)
            EXPERIMENTS+=("${ALL_D[@]}")
            ;;
        B1_clip_only|B2_map_only|B3_full|B4_full_token)
            EXPERIMENTS+=("$arg")
            ;;
        C1_channel_gate|C2_weighted_pool|C3_sigmoid_gate)
            EXPERIMENTS+=("$arg")
            ;;
        D_retrieval)
            EXPERIMENTS+=("$arg")
            ;;
        *)
            echo "Unknown experiment: $arg"
            exit 1
            ;;
    esac
done

# Remove duplicates
EXPERIMENTS=($(printf '%s\n' "${EXPERIMENTS[@]}" | sort -u))

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running ${#EXPERIMENTS[@]} experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

START_TIME=$(date +%s)
FAILED=()
PASSED=()

for exp in "${EXPERIMENTS[@]}"; do
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Running: ${exp}${NC}"
    echo "Command: python -m project.main experiment=${exp}"
    echo ""
    
    if python -m project.main experiment="${exp}"; then
        PASSED+=("$exp")
        echo -e "${GREEN}✓ ${exp} completed${NC}"
    else
        FAILED+=("$exp")
        echo -e "\033[0;31m✗ ${exp} failed${NC}"
    fi
    
    echo ""
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"

if [[ ${#PASSED[@]} -gt 0 ]]; then
    echo -e "${GREEN}Passed (${#PASSED[@]}):${NC}"
    for exp in "${PASSED[@]}"; do
        echo -e "  ${GREEN}✓${NC} $exp"
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo -e "\033[0;31mFailed (${#FAILED[@]}):${NC}"
    for exp in "${FAILED[@]}"; do
        echo -e "  \033[0;31m✗${NC} $exp"
    done
fi

echo ""
echo "Total time: $((ELAPSED / 60))m $((ELAPSED % 60))s"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi

exit 0

#!/usr/bin/env bash

# Usage:
#   Source this library in your bash scripts:
#     source scripts/bash_utils.sh
#
#   Then call the provided functions as needed.

# A guard to prevent multiple inclusions.
if [[ -n "${BASH_UTILS_LIB_INCLUDED:-}" ]]; then
    return
fi
BASH_UTILS_LIB_INCLUDED=1

# Enable safe bash options.
set -euo pipefail
IFS=$'\n\t'

#------------------------------------------------------------
# Function: print_on_main_processor
# Description:
#   Prints a message only if the current process is the main one.
#   It checks for an environment variable "RANK" and prints the message if RANK is 0.
#
# Parameters:
#   All parameters passed are printed as part of the message.
#------------------------------------------------------------
print_main() {
    local rank="${LOCAL_RANK:-0}"
    if [[ "$rank" -eq 0 ]]; then
        echo "[P0] $*"
    fi
}

#------------------------------------------------------------
# Function: run_command
# Description:
#   Executes a bash command
#
# Parameters:
#   A command passed as a string.
#
# Example:
#   run_python_command "python train.py"
#------------------------------------------------------------
run_command() {
    if [[ $# -eq 0 ]]; then
        echo "Error: No Python command provided."
        return 1
    fi

    local cmd="$*"
    print_main "Running :: $cmd"

    eval "$cmd"
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        print_main "Error: Command failed with exit code $exit_code"
        exit $exit_code
    fi
}

run_command_with_errors() {
    if [[ $# -eq 0 ]]; then
        echo "Error: No Python command provided."
        return 1
    fi

    local cmd="$*"
    print_main "Running :: $cmd"

    eval "$cmd"
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        print_main "Error: Command failed with exit code $exit_code"
    fi
}

# Convert config path to wandb run_name
config_path_to_run_name() {
  local raw="$1"
  local result="${raw#configs/}"        # Remove the prefix "configs/"
  result="${result%.yaml}"             # Remove the suffix ".yaml"
  echo "$result"
}

# Convert config path to checkpoint name
config_path_to_ckpt_name() {
  local run_name
  run_name=$(config_path_to_run_name "$1")
  result="${run_name//\//__}"            # Replace all "/" with "__"
  echo "$result"
}


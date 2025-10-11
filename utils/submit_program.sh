#!/bin/bash

# Usage: ./submit_program.sh path/to/binary

if [ -z "$1" ]; then
  echo "Usage: $0 path/to/binary"
  exit 1
fi

BIN_PATH=$1
BIN_NAME=$(basename "$BIN_PATH")
SCRIPT_NAME="job_${BIN_NAME}.sbatch"

cat > "$SCRIPT_NAME" <<EOF
#!/bin/bash
#SBATCH --job-name=${BIN_NAME}
#SBATCH --output=${BIN_NAME}.out
#SBATCH --error=${BIN_NAME}.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

cd gem5
./build/RISCV/gem5.opt configs/example/gem5_library/run_se.py "$BIN_PATH"
EOF

# Submit the generated script
sbatch "$SCRIPT_NAME"


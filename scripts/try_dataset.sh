PROJECT_PATH="/radraid/colivares/github_repos/svt"
DATA_PATH="/radraid/colivares/videos"
EXP_NAME="svt_test"

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=0

python3 /radraid/colivares/github_repos/svt/kinetics_sandbox.py \

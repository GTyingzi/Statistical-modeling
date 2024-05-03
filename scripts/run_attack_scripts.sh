#!/bin/bash
# 运行scripts下的所有脚本
chmod +x ./scripts/*.sh

./scripts/bert-softmax_fgm.sh
./scripts/bert-softmax_fgsm.sh
./scripts/bert-softmax_pgd.sh
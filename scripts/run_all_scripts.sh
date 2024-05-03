#!/bin/bash
# 运行scripts下的所有脚本
chmod +x ./scripts/*.sh

# 序列标注
./scripts/bert-softmax.sh
./scripts/bert-crf.sh
./scripts/bert-bilstm-crf.sh

# span标注
./scripts/bert-span.sh
./scripts/bert-globalpointer.sh
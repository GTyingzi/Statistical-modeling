#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/22 16:58 
# ide： PyCharm
from attack_module.attack_train import FGM, FGSM, PGD

def attack_model(attack_name, model, input_ids, attention_mask, token_type_ids, label_ids):
    if attack_name == "fgm":
        fgm = FGM(model=model)
        fgm.attack()
        loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
        loss.backward()
        fgm.restore()

    elif attack_name == "fgsm":
        fgsm = FGSM(model=model)
        fgsm.attack()
        loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
        loss.backward()
        fgsm.restore()

    elif attack_name == "pgd":
        pgd = PGD(model=model)
        pgd_k = 3
        pgd.backup_grad()
        for _t in range(pgd_k):
            pgd.attack(is_first_attack=(_t == 0))

            if _t != pgd_k - 1:
                model.zero_grad()
            else:
                pgd.restore_grad()

            loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
            loss.backward()
        pgd.restore()
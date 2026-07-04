python script/select_nosucess.py \
  --transfer-json transfer_result/transfer_torchvision_densenet121__to__bcos_densenet121.json \
  --attack-root /datastore/elo/khoatn/On-Adversarial-Bcos/attack_result \
  --epsilon 0.03 \
  --sample-size 100 \
  --seed 42 \
  --imagenet-val-dir /datastore/elo/quanphm/dataset/ImageNet1K/val \
  --annotations-file script/id_2_classname.json \
  --output localized/transfer_failed_100.json
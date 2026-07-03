# Init and Prediction model

import torch
from PIL import Image
from attack.util import (
    load_model,
    load_imagenet_categories,
    save_perturbation_image,
    save_rgb_image,
    save_explanation_rgba
)

model = load_model(
    model_type='bcosify', # ['torchvision', 'bcos', 'bcosify']
    model_name="simple_vit_b_patch16_224", # ['resnet50', 'densenet121', 'vgg16']
    device="cuda"
)
path_to_image = 'test_img/gecko.png'
image = Image.open(path_to_image)
image = model.transform.spatial_transform(image).unsqueeze(0).cuda()
image_input = model.transform.inverse_transform(image)



explain, logit = model.explain(image_input) # model(image_input) for prediction only: 
print("Logits shape:", logit.shape)
print("Predicted class:", logit.argmax(dim=1).item())
print("Explanation shape:", explain['explanation'].shape)

save_explanation_rgba(explain['explanation'], 'test_img/gecko_explanation.png')
print(image_input.shape) 
raise
# AttackModule
from attack.PGD import PGDAttack
from attack.SimBaAttack import SimBAAttack

attacker = PGDAttack(
    model=model,
    epsilon=0.03
)

adv_rgb, final_pred, success_step, history = attacker.solve(
    clean_rgb=image,
    original_class=logit.argmax(dim=1).item(),
    step_size=0.01,
    steps=100,
    target_class=None, # canbe,
    loss_fn=None # canbe, default is crossentropy
)

print("Orgiinal class:", logit.argmax(dim=1).item())
print("Adversarial class:", final_pred)
print("Attack success step:", success_step)

save_rgb_image(adv_rgb, 'test_img/gecko_adv.png')
save_rgb_image(image, 'test_img/gecko_input.png')

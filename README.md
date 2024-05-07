# Multi-label Human-Feedback-Boosted Generation of Histopathological Images
This project was part of our course DH602@ IIT Bombay.


Inspired from the paper, 
“Aligning synthetic medical images with clinical knowledge using human feedback” 
by S. Sun, G. Goldgof, A. Butte, and A. M. Alaa, in Advances in Neural Information Processing Systems, vol. 36, 2024, 

we present a more nuanced feedback scheme - to have a multiple labels as criterion for plausibility of generated image and fine-tune the diffusion model with the feedback of a reward model,
that is trained on the annotations by an expert pathologist.

Our directory structure is as follows:
1.  <b>Pre_training</b>: has all the files for training the diffusion model. train.py is used to train and generate.py used to run inference and generate images which were annotated by the Pathologist. 'training_epochs' subfolder has sample images for different epochs. 
2.  <b>Fine_tuning</b>:  has all the files utilised for fine-tuning the diffusion model.  'finetune_final99.py' was used to run the fine tuning. 'generate_from_finetuned.py' used to run inference and generate images for final validation.
3.  <b>Reward_model</b>: 'reward_model.py' is the complete file for training the reward model.
4.  <b>ablation_study_1</b>:
5.  <b>ablation_study_2</b>: 'reward_model.py' is used to train the reward model for binary classification of the images. 'finetune.py' is used to finetune the diffusion model using the original images and the images annotated by the reward model.
6.  <b>original_training_dataset</b>: contains the images from MedMNist. https://medmnist.com/

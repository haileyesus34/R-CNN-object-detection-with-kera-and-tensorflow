import os 

orig_base_path = 'raccons'
orig_images = os.path.sep.join([orig_base_path, 'images'])
orig_annots = os.path.sep.join([orig_base_path, 'annotations'])

base_path = 'dataset'
positive_path = os.path.sep.join([base_path, 'raccons'])
negative_path = os.path.sep.join([base_path, 'no_raccons'])

max_proposals = 2000
max_proposals_infer = 200 

max_positve = 30 
max_negative = 10 

input_dims = (224, 224)

model_path = 'raccon_detector.keras'
encoder_path = 'label_encoder.pickle'

min_proba= 0.99

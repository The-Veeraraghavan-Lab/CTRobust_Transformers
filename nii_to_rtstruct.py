import os
import SimpleITK as sitk
import numpy as np
from rt_utils import RTStructBuilder


roi_dict = {1 :'aorta', 2 :'brain', 3 :'heart_atrium_left', 4 :'heart_atrium_right', 5 :'heart_myocardium', 6 :'heart_ventricle_left', 7 :'heart_ventricle_right', 8 :'pancreas', 9 :'portal_vein_and_splenic_vein', 10 :'pulmonary_artery', 11 :'spleen', 12 :'stomach', 13 :'trachea', 14 :'adrenal_gland_left', 15 :'adrenal_gland_right', 16 :'inferior_vena_cava', 17 :'small_bowel', 18 :'duodenum', 19 :'gallbladder', 20 :'colon', 21 :'esophagus', 22 :'kidney_left', 23 :'kidney_right', 24 :'liver', 25 :'urinary_bladder'}

niimask = '/scratch/nifti/tep_result.nii.gz'

dcm_in = '/scratch/dicom'
dcm_out = '/scratch/output'

os.makedirs(dcm_out,exist_ok=True)

#convert RT struct
label_img = sitk.ReadImage(niimask)

rtstructout = os.path.join(dcm_out,'RTSTRUCT.dcm')
rtstruct = RTStructBuilder.create_new(dicom_series_path = dcm_in)

for roi in roi_dict:
	roi_arr = sitk.GetArrayFromImage(label_img) == roi
	rtstruct.add_roi(mask = np.transpose(roi_arr,(1,2,0)), name=roi_dict[roi])

rtstruct.save(rtstructout)

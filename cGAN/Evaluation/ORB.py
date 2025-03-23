from skimage.metrics import structural_similarity
import cv2
import glob
import os
import csv

def orb_sim(img1, img2):

  orb = cv2.ORB_create()

 
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
 
  matches = bf.match(desc_a, desc_b)
 
  similar_regions = [i for i in matches if i.distance < 50]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)

y_gen_folder = '/Users/deniztekin/Documents/Uni/Masterarbeit/inference/inferenz/Batch8/50/inference/y_gen'

label_folder = '/Users/deniztekin/Documents/Uni/Masterarbeit/inference/inferenz/Batch8/50/inference/label'

def get_label_file(y_gen_file):
    # Extract the epoch and batch from the y_gen file name
    parts = os.path.basename(y_gen_file).split('_')
    batch = parts[4].split('.')[0]  # Extract batch from y_gen file name
    
    # Generate corresponding label file name
    label_pattern = os.path.join(label_folder, f'label_*_batch_{batch}.tiff')
    
    # Find all files that match the pattern
    matching_files = glob.glob(label_pattern)
    
    # Return the first match if any file exists, otherwise return None
    return matching_files[0] if matching_files else None

results = []

y_gen_files = glob.glob(os.path.join(y_gen_folder, 'y_gen_*_batch_*.tiff'))

for y_gen_file in y_gen_files:
    
    parts = os.path.basename(y_gen_file).split('_')
    epoch = parts[2]  # Epoch number from y_gen file name
    batch = parts[4].split('.')[0]  # Batch number from y_gen file name
    
    
    # Load the y_gen image
    y_gen_img = cv2.imread(y_gen_file, 0)
    
    # Get the corresponding label image
    label_file = get_label_file(y_gen_file)
    if label_file:
        label_img = cv2.imread(label_file, 0)
        
        # Calculate the ORB similarity using your existing `orb_sim` function
        orb_similarity = orb_sim(y_gen_img, label_img)
        print(f'ORB similarity between {y_gen_file} and {label_file}: {orb_similarity}')
        
        results.append([epoch, batch, orb_similarity])
    else:
        print(f'No corresponding label found for {y_gen_file}')

results = [(int(epoch), int(batch), similarity) for epoch, batch, similarity in results]
results.sort(key=lambda x: (x[0], x[1]))



csv_file_path = '/Users/deniztekin/Documents/Uni/Masterarbeit/inference/inferenz/Batch8/50/orb_similarity_scores.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Batch", "ORB_Similarity"])  # Write header
    writer.writerows(results)  # Write data rows

print(f'ORB similarity scores saved to {csv_file_path}')

from skimage.metrics import structural_similarity
import cv2
import glob
import os
import csv

def structural_sim(img1, img2):

  sim, diff = structural_similarity(img1, img2, full=True)
  return sim

y_gen_folder = '/Users/deniztekin/Documents/Uni/Masterarbeit/y_gen'

label_folder = '/Users/deniztekin/Documents/Uni/Masterarbeit/label'

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
        SSIM = structural_sim(y_gen_img, label_img)
        print(f'SSIM between {y_gen_file} and {label_file}: {SSIM}')
        
        results.append([epoch, batch, SSIM])
    else:
        print(f'No corresponding label found for {y_gen_file}')


results = [(int(epoch), int(batch), similarity) for epoch, batch, similarity in results]
results.sort(key=lambda x: (x[0], x[1]))

csv_file_path = 'ssim_similarity_scores.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Batch", "SSIM_Similarity"])  # Write header
    writer.writerows(results)  # Write data rows

print(f'SSIM similarity scores saved to {csv_file_path}')








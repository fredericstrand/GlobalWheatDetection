import os
import cv2
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

def count_wheat_heads(input_folder, model_path="best.pt", output_folder="wheat_heads_detected", confidence_threshold=0.2):
    """
    Process images in a folder to count wheat heads using YOLOv8
    
    Args:
        input_folder (str): Path to folder containing plot images
        model_path (str): Path to YOLOv8 model file (best.pt)
        output_folder (str): Name for output folder with annotated images
        confidence_threshold (float): Minimum confidence for detections
    """
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    
    # Create output directory
    input_path = Path(input_folder)
    output_path = input_path.parent / output_folder
    output_path.mkdir(exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Get all image files in the input folder and subfolders
    image_files = []
    
    # First check direct files in the main folder
    direct_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    image_files.extend(direct_files)
    
    # Then check subfolders for image files
    for subfolder in input_path.iterdir():
        if subfolder.is_dir():
            subfolder_files = [f for f in subfolder.iterdir() 
                             if f.is_file() and f.suffix.lower() in image_extensions]
            image_files.extend(subfolder_files)
            print(f"Found {len(subfolder_files)} images in {subfolder.name}")
    
    if not image_files:
        print(f"No image files found in {input_folder} or its subfolders")
        print("Contents of the folder:")
        for item in input_path.iterdir():
            if item.is_dir():
                print(f"  - {item.name} (folder)")
                # Show contents of subfolders
                subfolder_contents = list(item.iterdir())
                if subfolder_contents:
                    print(f"    Contents: {[f.name for f in subfolder_contents[:5]]}")
                    if len(subfolder_contents) > 5:
                        print(f"    ... and {len(subfolder_contents) - 5} more files")
            else:
                print(f"  - {item.name} (file)")
        return None
    
    print(f"Found {len(image_files)} images to process")
    
    # Results storage
    results_data = []
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {image_file.name}")
        
        try:
            # Run inference
            results = model(str(image_file), conf=confidence_threshold)
            
            # Get detection results
            detections = results[0]
            wheat_head_count = len(detections.boxes) if detections.boxes is not None else 0
            
            # Load original image for annotation
            image = cv2.imread(str(image_file))
            annotated_image = image.copy()
            
            # Draw bounding boxes on the image
            if detections.boxes is not None:
                boxes = detections.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
                confidences = detections.boxes.conf.cpu().numpy()  # Get confidence scores
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add confidence label
                    label = f'Wheat: {conf:.2f}'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1-label_size[1]-10), 
                                (x1+label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_image, label, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add count text to image
            count_text = f'Wheat Heads: {wheat_head_count}'
            cv2.putText(annotated_image, count_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Save annotated image with plot name in filename
            plot_name = image_file.parent.name if image_file.parent.name != "Counting" else image_file.stem
            output_image_path = output_path / f"annotated_{plot_name}_{image_file.name}"
            cv2.imwrite(str(output_image_path), annotated_image)
            
            # Store results
            plot_name = image_file.parent.name if image_file.parent.name != "Counting" else image_file.stem
            results_data.append({
                'Plot_Name': plot_name,
                'Image_File': image_file.name,
                'Wheat_Head_Count': wheat_head_count,
                'Output_Image': f"annotated_{plot_name}_{image_file.name}",
                'Source_Folder': image_file.parent.name
            })
            
            print(f"  -> Found {wheat_head_count} wheat heads")
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
            plot_name = image_file.parent.name if image_file.parent.name != "Counting" else image_file.stem
            results_data.append({
                'Plot_Name': plot_name,
                'Image_File': image_file.name,
                'Wheat_Head_Count': 'Error',
                'Output_Image': 'Error',
                'Source_Folder': image_file.parent.name,
                'Error': str(e)
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    csv_path = output_path / 'wheat_head_counts.csv'
    results_df.to_csv(csv_path, index=False)
    
    # Print summary
    print(f"\n{'='*50}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*50}")
    print(f"Processed images: {len(image_files)}")
    print(f"Output folder: {output_path}")
    print(f"Results saved to: {csv_path}")
    
    # Display summary statistics
    successful_counts = [r['Wheat_Head_Count'] for r in results_data if isinstance(r['Wheat_Head_Count'], int)]
    if successful_counts:
        print(f"\nSUMMARY STATISTICS:")
        print(f"Total wheat heads detected: {sum(successful_counts)}")
        print(f"Average per plot: {sum(successful_counts)/len(successful_counts):.1f}")
        print(f"Min count: {min(successful_counts)}")
        print(f"Max count: {max(successful_counts)}")
    
    return results_df

# Example usage
if __name__ == "__main__":
    # Auto-detect the Counting folder in current directory
    current_dir = Path.cwd()
    input_folder = current_dir / "Counting"
    model_path = current_dir / "best.pt"
    
    print(f"Current directory: {current_dir}")
    print(f"Looking for Counting folder at: {input_folder}")
    print(f"Looking for model at: {model_path}")
    
    # Check if paths exist
    if not input_folder.exists():
        print(f"\nERROR: Counting folder not found at {input_folder}")
        print("Available folders in current directory:")
        for item in current_dir.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        exit(1)
    
    if not model_path.exists():
        print(f"\nERROR: Model file not found at {model_path}")
        print("Available files in current directory:")
        for item in current_dir.iterdir():
            if item.is_file():
                print(f"  - {item.name}")
        exit(1)
    
    # Run the wheat head counting
    results = count_wheat_heads(
        input_folder=str(input_folder),
        model_path=str(model_path),
        confidence_threshold=0.5  # Adjust as needed
    )
    
    if results is not None:
        print("\nFirst few results:")
        print(results.head())
    else:
        print("No results to display.")
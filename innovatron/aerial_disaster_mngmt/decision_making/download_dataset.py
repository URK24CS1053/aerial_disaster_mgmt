"""
Download and process the UCI Heart Disease dataset
for the fusion engine training.
"""

import urllib.request
import csv
import os

def download_heart_disease_dataset():
    """Download Heart Disease dataset from UCI repository"""
    
    # UCI Heart Disease dataset URL (Cleveland database)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    dataset_path = os.path.join(os.path.dirname(__file__), "heart_disease_raw.csv")
    
    print(f"Downloading Heart Disease dataset from UCI...")
    try:
        urllib.request.urlretrieve(url, dataset_path)
        print(f"Downloaded to {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


def transform_to_sensor_data(raw_path, output_path):
    """Transform Heart Disease dataset to sensor fusion format"""
    
    # Heart Disease attributes:
    # 0: age, 1: sex, 2: cp (chest pain), 3: trestbps (resting BP), 
    # 4: chol (cholesterol), 5: fbs, 6: restecg, 7: thalach (max HR),
    # 8: exang (exercise induced angina), 9: oldpeak, 10: slope, 
    # 11: ca, 12: thal, 13: target (0=no disease, 1-4=disease)
    
    training_data = []
    
    try:
        with open(raw_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 14 or '?' in row:  # Skip incomplete rows
                    continue
                
                try:
                    age = float(row[0])
                    max_hr = float(row[7])  # Thalach - max heart rate
                    exang = int(float(row[8]))  # Exercise induced angina (0/1)
                    target = int(float(row[13]))  # Disease presence
                    
                    # Map to sensor fusion features:
                    # human_detected: 1 if record exists, 0 otherwise
                    human_detected = 1
                    
                    # motion_level: based on max heart rate (0=NONE, 1=LOW, 2=HIGH)
                    if max_hr > 150:
                        motion_level = 2  # HIGH motion
                    elif max_hr > 100:
                        motion_level = 1  # LOW motion
                    else:
                        motion_level = 0  # NONE
                    
                    # heat_presence: based on resting BP (0=LOW, 1=NORMAL)
                    rest_bp = float(row[3])
                    heat_presence = 1 if rest_bp >= 120 else 0
                    
                    # breathing_detected: based on exercise induced angina (1=YES, 0=NO)
                    breathing_detected = 1 if exang == 0 else 0
                    
                    # confidence_level: map disease presence to confidence
                    # target=0 (no disease, healthy) = HIGH confidence
                    # target=1,2 (mild disease) = MEDIUM confidence  
                    # target=3,4 (severe disease) = LOW confidence
                    if target == 0:
                        confidence = 2  # HIGH - healthy, stable
                    elif target in [1, 2]:
                        confidence = 1  # MEDIUM - some issues
                    else:
                        confidence = 0  # LOW - severe issues
                    
                    training_data.append([
                        human_detected,
                        motion_level,
                        heat_presence,
                        breathing_detected,
                        confidence
                    ])
                except (ValueError, IndexError):
                    continue
    
    except Exception as e:
        print(f"Error reading raw data: {e}")
        return False
    
    # Write transformed data
    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['human_detected', 'motion_level', 'heat_presence', 'breathing_detected', 'confidence_level'])
            writer.writerows(training_data)
        
        print(f"Transformed {len(training_data)} records to {output_path}")
        return True
    except Exception as e:
        print(f"Error writing transformed data: {e}")
        return False


if __name__ == "__main__":
    raw_path = download_heart_disease_dataset()
    if raw_path:
        output_path = os.path.join(os.path.dirname(__file__), "sensor_training_data.csv")
        if transform_to_sensor_data(raw_path, output_path):
            print("\n✓ Dataset ready! You can now run fusion_engine.py")
            # Optionally remove raw file
            try:
                os.remove(raw_path)
                print("Cleaned up temporary files")
            except:
                pass
